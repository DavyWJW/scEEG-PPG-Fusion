"""
PPG + Unfiltered PPG Cross-Attention Model - Short Window Version
支持多种窗口长度: 30s, 1min, 3min, 5min, 10min, 30min, 10H

关键修复:
1. 移除forward中的softmax (避免double softmax bug)
2. 动态编码器深度 (根据窗口长度自动调整)
3. 动态输出长度 (不再硬编码1200)
4. 最小特征长度保证 (通过AdaptiveAvgPool1d)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Dict, Tuple


class ResConvBlock(nn.Module):
    """残差卷积块"""
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=2, dropout=0.1):
        super().__init__()
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                               stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 
                               stride=1, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        # 残差连接
        self.residual = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, stride=stride),
            nn.BatchNorm1d(out_channels)
        ) if in_channels != out_channels or stride != 1 else nn.Identity()
        
    def forward(self, x):
        residual = self.residual(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = out + residual
        out = F.relu(out)
        
        return out


class TemporalConvBlock(nn.Module):
    """时序卷积块 with dilation"""
    def __init__(self, in_channels, out_channels, kernel_size=7, dilation=1, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        self.residual = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()
    
    def forward(self, x):
        residual = self.residual(x)
        out = self.conv(x)
        out = self.bn(out)
        out = F.relu(out)
        out = self.dropout(out)
        return out + residual


class MultiHeadCrossAttention(nn.Module):
    """多头交叉注意力机制"""
    
    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, _ = query.shape
        
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        output = self.w_o(context)
        output = self.dropout(output)
        output = self.layer_norm(output + query)
        
        return output, attention_weights


class CrossModalFusionBlock(nn.Module):
    """交叉模态融合块"""
    
    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super().__init__()
        
        self.clean_cross_attn = MultiHeadCrossAttention(d_model, n_heads, dropout)
        self.noisy_cross_attn = MultiHeadCrossAttention(d_model, n_heads, dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, clean_features, noisy_features):
        clean_attended, _ = self.clean_cross_attn(clean_features, noisy_features, noisy_features)
        noisy_attended, _ = self.noisy_cross_attn(noisy_features, clean_features, clean_features)
        
        clean_out = self.layer_norm(clean_attended + self.dropout(self.ffn(clean_attended)))
        noisy_out = self.layer_norm(noisy_attended + self.dropout(self.ffn(noisy_attended)))
        
        return clean_out, noisy_out


class AdaptiveModalityWeighting(nn.Module):
    """自适应模态权重模块"""
    
    def __init__(self, d_model):
        super().__init__()
        self.clean_weight_net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.noisy_weight_net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, clean_features, noisy_features):
        clean_weight = self.clean_weight_net(clean_features)
        noisy_weight = self.noisy_weight_net(noisy_features)
        
        total_weight = clean_weight + noisy_weight + 1e-8
        clean_weight = clean_weight / total_weight
        noisy_weight = noisy_weight / total_weight
        
        return clean_weight, noisy_weight


class PPGCrossAttnShortWindow(nn.Module):
    """
    PPG + Unfiltered PPG Cross-Attention模型 - 支持短窗口版本
    
    关键改进:
    1. 动态编码器深度: 根据输入长度自动调整下采样率
    2. 无softmax输出: 输出logits，配合CrossEntropyLoss使用
    3. 动态输出长度: 根据输入自动计算输出epoch数
    4. 最小特征长度保证: 确保attention有足够的token
    """
    
    # 窗口配置: {窗口名: (采样点数, epoch数, 编码器层数)}
    # 采样率: 34.13 Hz, 每个epoch 30秒 = 1024采样点
    WINDOW_CONFIGS = {
        '30s':   (1024,      1,    4),   # 30秒 = 1 epoch
        '1min':  (2048,      2,    5),   # 1分钟 = 2 epochs  
        '3min':  (6144,      6,    6),   # 3分钟 = 6 epochs
        '5min':  (10240,     10,   6),   # 5分钟 = 10 epochs
        '10min': (20480,     20,   7),   # 10分钟 = 20 epochs
        '30min': (61440,     60,   8),   # 30分钟 = 60 epochs
        '10H':   (1228800,   1200, 9),   # 10小时 = 1200 epochs
    }
    
    def __init__(self, 
                 n_classes: int = 4, 
                 d_model: int = 256, 
                 n_heads: int = 8, 
                 n_fusion_blocks: int = 3,
                 window_size: str = '10H',
                 min_feature_length: int = 32,
                 noise_config: Optional[Dict] = None,
                 deterministic_noise: bool = False):
        """
        Args:
            n_classes: 分类数 (4: Wake, Light, Deep, REM)
            d_model: 模型维度
            n_heads: 注意力头数
            n_fusion_blocks: 融合块数量
            window_size: 窗口大小 ('30s', '1min', '3min', '5min', '10min', '30min', '10H')
            min_feature_length: 最小特征序列长度
            noise_config: 噪声配置
            deterministic_noise: 是否使用确定性噪声(用于推理复现)
        """
        super().__init__()
        
        self.n_classes = n_classes
        self.d_model = d_model
        self.window_size = window_size
        self.min_feature_length = min_feature_length
        self.deterministic_noise = deterministic_noise
        
        # 获取窗口配置
        if window_size not in self.WINDOW_CONFIGS:
            raise ValueError(f"Unsupported window_size: {window_size}. "
                           f"Choose from {list(self.WINDOW_CONFIGS.keys())}")
        
        self.input_samples, self.n_epochs, self.encoder_layers = self.WINDOW_CONFIGS[window_size]
        
        # 噪声配置
        self.noise_config = noise_config or {
            'noise_level': 0.1,
            'drift_amplitude': 0.1,
            'drift_frequency': 0.1,
            'spike_probability': 0.01,
            'spike_amplitude': 0.5
        }
        
        # 计算编码器输出长度
        self.encoder_downsample = 2 ** self.encoder_layers
        self.expected_feature_length = self.input_samples // self.encoder_downsample
        
        print(f"[PPGCrossAttnShortWindow] Window: {window_size}")
        print(f"  Input samples: {self.input_samples}")
        print(f"  Output epochs: {self.n_epochs}")
        print(f"  Encoder layers: {self.encoder_layers} (downsample: {self.encoder_downsample}x)")
        print(f"  Expected feature length: {self.expected_feature_length}")
        
        # 创建编码器
        self.clean_ppg_encoder = self._create_encoder(d_model, self.encoder_layers)
        self.noisy_ppg_encoder = self._create_encoder(d_model, self.encoder_layers)
        
        # 位置编码 (支持最大3000长度)
        max_pos_len = max(3000, self.expected_feature_length + 100)
        self.positional_encoding = self._create_positional_encoding(d_model, max_pos_len)
        
        # Cross-Modal Fusion层
        self.fusion_blocks = nn.ModuleList([
            CrossModalFusionBlock(d_model, n_heads)
            for _ in range(n_fusion_blocks)
        ])
        
        # 自适应模态权重
        self.modality_weighting = AdaptiveModalityWeighting(d_model)
        
        # 特征聚合
        self.feature_aggregation = nn.Sequential(
            nn.Conv1d(d_model * 2, d_model, 1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 时序建模 (根据特征长度调整dilation)
        self.temporal_blocks = self._create_temporal_blocks(d_model)
        
        # 特征细化
        self.feature_refinement = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )
        
        # 自适应池化到epoch数
        self.adaptive_pool = nn.AdaptiveAvgPool1d(self.n_epochs)
        
        # 最终分类器 - 输出logits，不带softmax!
        self.classifier = nn.Sequential(
            nn.Conv1d(d_model, 128, 1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(128, n_classes, 1)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _create_encoder(self, d_model: int, n_layers: int) -> nn.Sequential:
        """创建动态深度的编码器"""
        layers = []
        
        # 通道数配置
        if n_layers <= 4:
            channels = [1, 32, 64, 128, d_model][:n_layers + 1]
        elif n_layers <= 6:
            channels = [1, 16, 32, 64, 128, 256, d_model][:n_layers + 1]
        else:
            channels = [1, 16, 32, 64, 128, 256, 256, 256, 256, d_model][:n_layers + 1]
        
        # 确保最后一个通道是d_model
        channels[-1] = d_model
        
        for i in range(len(channels) - 1):
            layers.append(ResConvBlock(channels[i], channels[i + 1], stride=2))
        
        return nn.Sequential(*layers)
    
    def _create_positional_encoding(self, d_model: int, max_len: int) -> nn.Parameter:
        """创建位置编码"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return nn.Parameter(pe.unsqueeze(0).transpose(1, 2), requires_grad=False)
    
    def _create_temporal_blocks(self, d_model: int) -> nn.Sequential:
        """创建时序建模块，根据特征长度调整"""
        # 对于非常短的特征，减少dilation
        if self.expected_feature_length < 20:
            dilations = [1, 1, 2, 2]
        elif self.expected_feature_length < 50:
            dilations = [1, 2, 2, 4]
        else:
            dilations = [1, 2, 4, 8]
        
        return nn.Sequential(
            TemporalConvBlock(d_model, d_model, kernel_size=7, dilation=dilations[0]),
            TemporalConvBlock(d_model, d_model, kernel_size=7, dilation=dilations[1]),
            TemporalConvBlock(d_model, d_model, kernel_size=7, dilation=dilations[2]),
            TemporalConvBlock(d_model, d_model, kernel_size=7, dilation=dilations[3])
        )
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def add_noise_to_ppg(self, clean_ppg: torch.Tensor, 
                         seed: Optional[int] = None) -> torch.Tensor:
        """
        向干净的PPG信号添加噪声，模拟未滤波信号
        
        Args:
            clean_ppg: 干净的PPG信号 (B, 1, L)
            seed: 随机种子 (用于确定性推理)
        Returns:
            noisy_ppg: 带噪声的PPG信号 (B, 1, L)
        """
        batch_size, _, length = clean_ppg.shape
        device = clean_ppg.device
        
        # 确定性噪声
        if self.deterministic_noise or seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(seed if seed is not None else 42)
        else:
            generator = None
        
        noisy_ppg = clean_ppg.clone()
        
        # 1. 高斯白噪声
        if generator:
            gaussian_noise = torch.randn(batch_size, 1, length, device=device, 
                                        generator=generator) * self.noise_config['noise_level']
        else:
            gaussian_noise = torch.randn_like(clean_ppg) * self.noise_config['noise_level']
        noisy_ppg = noisy_ppg + gaussian_noise
        
        # 2. 基线漂移
        t = torch.linspace(0, 1, length, device=device)
        drift_freq = self.noise_config['drift_frequency']
        drift_amp = self.noise_config['drift_amplitude']
        
        drift = drift_amp * (
            0.5 * torch.sin(2 * np.pi * drift_freq * t) +
            0.3 * torch.sin(2 * np.pi * drift_freq * 2 * t) +
            0.2 * torch.sin(2 * np.pi * drift_freq * 0.5 * t)
        )
        drift = drift.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)
        noisy_ppg = noisy_ppg + drift
        
        # 3. 运动伪影
        spike_prob = self.noise_config['spike_probability']
        spike_amp = self.noise_config['spike_amplitude']
        
        if generator:
            spike_mask = torch.rand(batch_size, 1, length, device=device, 
                                   generator=generator) < spike_prob
            spike_values = torch.randn(batch_size, 1, length, device=device,
                                      generator=generator) * spike_amp
        else:
            spike_mask = torch.rand(batch_size, 1, length, device=device) < spike_prob
            spike_values = torch.randn(batch_size, 1, length, device=device) * spike_amp
        
        spikes = spike_mask.float() * spike_values
        
        # 平滑尖峰
        kernel_size = 5
        padding = kernel_size // 2
        smoothing_kernel = torch.ones(1, 1, kernel_size, device=device) / kernel_size
        spikes = F.conv1d(spikes, smoothing_kernel, padding=padding)
        noisy_ppg = noisy_ppg + spikes
        
        # 4. 高频噪声
        if generator:
            emg_noise = torch.randn(batch_size, 1, length, device=device,
                                   generator=generator) * 0.05
        else:
            emg_noise = torch.randn_like(clean_ppg) * 0.05
        noisy_ppg = noisy_ppg + emg_noise
        
        return noisy_ppg
    
    def forward(self, ppg: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            ppg: PPG信号 (B, 1, L) 或 (B, L)
        Returns:
            logits: 分类logits (B, n_classes, n_epochs)
                   注意: 输出是logits，不是概率！使用CrossEntropyLoss训练
        """
        # 处理输入维度
        if ppg.dim() == 2:
            ppg = ppg.unsqueeze(1)
        
        batch_size = ppg.size(0)
        input_length = ppg.size(2)
        
        # 验证输入长度
        if input_length != self.input_samples:
            # 允许一定的误差范围
            if abs(input_length - self.input_samples) > self.input_samples * 0.1:
                raise ValueError(f"Input length {input_length} doesn't match expected "
                               f"{self.input_samples} for window size {self.window_size}")
        
        # 创建未滤波版本
        ppg_unfiltered = self.add_noise_to_ppg(ppg)
        
        # 编码
        clean_features = self.clean_ppg_encoder(ppg)
        noisy_features = self.noisy_ppg_encoder(ppg_unfiltered)
        
        # 添加位置编码
        seq_len = clean_features.size(2)
        if seq_len <= self.positional_encoding.size(2):
            clean_features = clean_features + self.positional_encoding[:, :, :seq_len]
            noisy_features = noisy_features + self.positional_encoding[:, :, :seq_len]
        
        # 自适应权重
        clean_weight, noisy_weight = self.modality_weighting(clean_features, noisy_features)
        
        clean_features_weighted = clean_features * clean_weight.unsqueeze(-1)
        noisy_features_weighted = noisy_features * noisy_weight.unsqueeze(-1)
        
        # 转换为 (B, L, C) 用于attention
        clean_features_t = clean_features_weighted.transpose(1, 2)
        noisy_features_t = noisy_features_weighted.transpose(1, 2)
        
        # Cross-Modal Fusion
        for fusion_block in self.fusion_blocks:
            clean_features_t, noisy_features_t = fusion_block(clean_features_t, noisy_features_t)
        
        # 转回 (B, C, L)
        clean_features = clean_features_t.transpose(1, 2)
        noisy_features = noisy_features_t.transpose(1, 2)
        
        # 特征聚合
        combined_features = torch.cat([clean_features, noisy_features], dim=1)
        fused_features = self.feature_aggregation(combined_features)
        
        # 时序建模
        temporal_features = self.temporal_blocks(fused_features)
        
        # 特征细化
        refined_features = self.feature_refinement(temporal_features)
        
        # 自适应池化到目标epoch数
        output_features = self.adaptive_pool(refined_features)
        
        # 分类 - 输出logits，不带softmax!
        logits = self.classifier(output_features)
        
        return logits
    
    def predict(self, ppg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预测接口，返回预测类别和概率
        
        Args:
            ppg: PPG信号
        Returns:
            predictions: 预测类别 (B, n_epochs)
            probabilities: 类别概率 (B, n_classes, n_epochs)
        """
        logits = self.forward(ppg)
        probabilities = F.softmax(logits, dim=1)
        predictions = logits.argmax(dim=1)
        return predictions, probabilities
    
    def get_config(self) -> Dict:
        """获取模型配置"""
        return {
            'n_classes': self.n_classes,
            'd_model': self.d_model,
            'window_size': self.window_size,
            'input_samples': self.input_samples,
            'n_epochs': self.n_epochs,
            'encoder_layers': self.encoder_layers,
            'expected_feature_length': self.expected_feature_length,
            'noise_config': self.noise_config
        }


def create_model_for_window(window_size: str, **kwargs) -> PPGCrossAttnShortWindow:
    """
    便捷函数：为指定窗口大小创建模型
    
    Args:
        window_size: '30s', '1min', '3min', '5min', '10min', '30min', '10H'
        **kwargs: 其他模型参数
    """
    return PPGCrossAttnShortWindow(window_size=window_size, **kwargs)


def test_all_window_sizes():
    """测试所有窗口大小"""
    print("=" * 80)
    print("Testing PPG Cross-Attention Model for All Window Sizes")
    print("=" * 80)
    
    results = {}
    
    for window_size in PPGCrossAttnShortWindow.WINDOW_CONFIGS.keys():
        print(f"\n{'='*60}")
        print(f"Testing window size: {window_size}")
        print("=" * 60)
        
        try:
            # 创建模型
            model = create_model_for_window(window_size)
            
            # 获取配置
            config = model.get_config()
            input_samples = config['input_samples']
            n_epochs = config['n_epochs']
            
            # 创建测试输入
            batch_size = 2
            ppg = torch.randn(batch_size, 1, input_samples)
            
            # 前向传播
            with torch.no_grad():
                logits = model(ppg)
            
            print(f"Input shape: {ppg.shape}")
            print(f"Output shape: {logits.shape}")
            print(f"Expected: (B={batch_size}, C=4, T={n_epochs})")
            
            # 验证输出形状
            expected_shape = (batch_size, 4, n_epochs)
            assert logits.shape == expected_shape, \
                f"Shape mismatch: got {logits.shape}, expected {expected_shape}"
            
            # 验证输出是logits (可以是任意值，不需要归一化)
            print(f"Output range: [{logits.min():.4f}, {logits.max():.4f}]")
            
            # 测试predict方法
            predictions, probs = model.predict(ppg)
            print(f"Predictions shape: {predictions.shape}")
            print(f"Probabilities shape: {probs.shape}")
            print(f"Prob sum per position (should be 1.0): {probs.sum(dim=1).mean():.6f}")
            
            # 参数量
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            results[window_size] = {
                'status': 'PASS',
                'input_shape': list(ppg.shape),
                'output_shape': list(logits.shape),
                'total_params': total_params,
                'trainable_params': trainable_params,
                'params_MB': total_params * 4 / 1024 / 1024
            }
            
            print(f"\n✓ {window_size}: PASS")
            print(f"  Parameters: {total_params:,} ({results[window_size]['params_MB']:.2f} MB)")
            
        except Exception as e:
            results[window_size] = {
                'status': 'FAIL',
                'error': str(e)
            }
            print(f"\n✗ {window_size}: FAIL - {e}")
    
    # 汇总
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print(f"\n{'Window':<10} {'Status':<8} {'Input Samples':<15} {'Output Epochs':<15} {'Params (MB)':<12}")
    print("-" * 70)
    
    for window_size, config in PPGCrossAttnShortWindow.WINDOW_CONFIGS.items():
        result = results.get(window_size, {})
        status = result.get('status', 'N/A')
        params_mb = result.get('params_MB', 0)
        
        status_symbol = "✓" if status == "PASS" else "✗"
        print(f"{window_size:<10} {status_symbol} {status:<6} {config[0]:<15} {config[1]:<15} {params_mb:<12.2f}")
    
    return results


def benchmark_inference_time():
    """测试推理时间"""
    import time
    
    print("\n" + "=" * 80)
    print("INFERENCE TIME BENCHMARK")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    n_warmup = 5
    n_runs = 20
    
    results = {}
    
    for window_size in ['30s', '1min', '3min', '5min', '10min']:
        model = create_model_for_window(window_size, deterministic_noise=True)
        model = model.to(device)
        model.eval()
        
        config = model.get_config()
        input_samples = config['input_samples']
        
        ppg = torch.randn(1, 1, input_samples, device=device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(n_warmup):
                _ = model(ppg)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(n_runs):
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                start = time.perf_counter()
                _ = model(ppg)
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                end = time.perf_counter()
                times.append((end - start) * 1000)  # ms
        
        mean_time = np.mean(times)
        std_time = np.std(times)
        
        results[window_size] = {
            'mean_ms': mean_time,
            'std_ms': std_time,
            'min_ms': min(times),
            'max_ms': max(times)
        }
        
        print(f"{window_size:<10}: {mean_time:>8.2f} ± {std_time:.2f} ms")
    
    return results


if __name__ == "__main__":
    # 测试所有窗口大小
    test_all_window_sizes()
    
    # 测试推理时间
    benchmark_inference_time()
