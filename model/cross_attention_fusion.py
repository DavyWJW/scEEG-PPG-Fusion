"""
EEG-PPG Cross-Attention Fusion Model
双向Cross-Attention融合，冻结Encoder只训练融合层

配置:
- 窗口长度: 3分钟 (6个epochs)
- 融合方向: 双向 (EEG ↔ PPG)
- 训练策略: 冻结Encoder，只训练Cross-Attention + 分类头
- 特征维度: 256
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class MultiHeadCrossAttention(nn.Module):
    """多头交叉注意力机制"""

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (batch, seq_len, d_model)
            key: (batch, seq_len, d_model)
            value: (batch, seq_len, d_model)
        Returns:
            output: (batch, seq_len, d_model)
            attention_weights: (batch, n_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = query.shape
        residual = query

        # 线性变换并分头
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 注意力计算
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 应用注意力权重
        context = torch.matmul(attention_weights, V)

        # 合并多头
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # 输出投影
        output = self.w_o(context)
        output = self.dropout(output)

        # 残差连接和层归一化
        output = self.layer_norm(output + residual)

        return output, attention_weights


class CrossModalFusionBlock(nn.Module):
    """双向交叉模态融合块"""

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        # EEG attend to PPG
        self.eeg_cross_attn = MultiHeadCrossAttention(d_model, n_heads, dropout)

        # PPG attend to EEG
        self.ppg_cross_attn = MultiHeadCrossAttention(d_model, n_heads, dropout)

        # 前馈网络
        self.eeg_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

        self.ppg_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

        self.eeg_norm = nn.LayerNorm(d_model)
        self.ppg_norm = nn.LayerNorm(d_model)

    def forward(self, eeg_features: torch.Tensor,
                ppg_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            eeg_features: (batch, seq_len, d_model)
            ppg_features: (batch, seq_len, d_model)
        Returns:
            eeg_out: (batch, seq_len, d_model)
            ppg_out: (batch, seq_len, d_model)
        """
        # EEG attend to PPG (用PPG信息增强EEG)
        eeg_attended, _ = self.eeg_cross_attn(eeg_features, ppg_features, ppg_features)

        # PPG attend to EEG (用EEG信息增强PPG)
        ppg_attended, _ = self.ppg_cross_attn(ppg_features, eeg_features, eeg_features)

        # FFN
        eeg_out = self.eeg_norm(eeg_attended + self.eeg_ffn(eeg_attended))
        ppg_out = self.ppg_norm(ppg_attended + self.ppg_ffn(ppg_attended))

        return eeg_out, ppg_out


class EEGPPGCrossAttentionFusion(nn.Module):
    """
    EEG-PPG Cross-Attention Fusion Model

    冻结预训练的EEG和PPG Encoder，只训练Cross-Attention融合层和分类头
    """

    def __init__(self,
                 eeg_model: nn.Module,
                 ppg_model: nn.Module,
                 d_model: int = 256,
                 n_heads: int = 8,
                 n_fusion_blocks: int = 2,
                 n_classes: int = 4,
                 dropout: float = 0.1,
                 freeze_encoders: bool = True):
        super().__init__()

        self.d_model = d_model
        self.n_classes = n_classes

        # 保存原模型（用于提取特征）
        self.eeg_model = eeg_model
        self.ppg_model = ppg_model

        # 冻结Encoder
        if freeze_encoders:
            self._freeze_encoders()

        # Cross-Modal Fusion层
        self.fusion_blocks = nn.ModuleList([
            CrossModalFusionBlock(d_model, n_heads, dropout)
            for _ in range(n_fusion_blocks)
        ])

        # 特征融合方式: Concatenation后投影
        self.fusion_projection = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes)
        )

        # 初始化新增层的权重
        self._init_weights()

    def _freeze_encoders(self):
        """冻结EEG和PPG的Encoder参数"""
        # 冻结EEG模型的所有参数
        for param in self.eeg_model.parameters():
            param.requires_grad = False

        # 冻结PPG模型的所有参数
        for param in self.ppg_model.parameters():
            param.requires_grad = False

        print("Encoders frozen. Only fusion layers and classifier will be trained.")

    def _init_weights(self):
        """初始化融合层和分类头的权重"""
        for module in [self.fusion_blocks, self.fusion_projection, self.classifier]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    def extract_eeg_features(self, eeg_input: torch.Tensor) -> torch.Tensor:
        """
        提取EEG特征（使用冻结的Encoder）

        Args:
            eeg_input: (batch, n_epochs, signal_length) = (batch, 6, 3000)
        Returns:
            features: (batch, n_epochs, d_model) = (batch, 6, 256)
        """
        batch_size, num_epochs, signal_length = eeg_input.shape

        # 使用EEG模型的MRCNN和TCE提取特征
        x_reshaped = eeg_input.view(batch_size * num_epochs, 1, signal_length)

        with torch.no_grad():
            # MRCNN特征提取
            epoch_features = self.eeg_model.mrcnn(x_reshaped)
            # TCE注意力
            epoch_features = self.eeg_model.tce(epoch_features)
            # 展平
            epoch_features = epoch_features.contiguous().view(batch_size * num_epochs, -1)
            # 特征压缩
            epoch_features = self.eeg_model.feature_compress(epoch_features)

        # 重塑为 (batch, n_epochs, d_model)
        features = epoch_features.view(batch_size, num_epochs, -1)

        return features

    def extract_ppg_features(self, ppg_input: torch.Tensor) -> torch.Tensor:
        """
        提取PPG特征（使用冻结的Encoder）

        Args:
            ppg_input: (batch, signal_length) = (batch, 6144) for 3min
        Returns:
            features: (batch, n_epochs, d_model) = (batch, 6, 256)
        """
        batch_size = ppg_input.size(0)

        # 添加通道维度
        if ppg_input.dim() == 2:
            ppg_input = ppg_input.unsqueeze(1)  # (batch, 1, signal_length)

        with torch.no_grad():
            # 创建噪声版本
            ppg_noisy = self.ppg_model.add_noise_to_ppg(ppg_input)

            # 编码两个流
            clean_features = self.ppg_model.clean_ppg_encoder(ppg_input)
            noisy_features = self.ppg_model.noisy_ppg_encoder(ppg_noisy)

            # 添加位置编码
            seq_len = clean_features.size(2)
            if seq_len <= self.ppg_model.positional_encoding.size(2):
                clean_features = clean_features + self.ppg_model.positional_encoding[:, :, :seq_len]
                noisy_features = noisy_features + self.ppg_model.positional_encoding[:, :, :seq_len]

            # 获取自适应权重
            clean_weight, noisy_weight = self.ppg_model.modality_weighting(clean_features, noisy_features)

            # 应用权重
            clean_features = clean_features * clean_weight.unsqueeze(-1)
            noisy_features = noisy_features * noisy_weight.unsqueeze(-1)

            # 转换格式用于attention
            clean_features_t = clean_features.transpose(1, 2)
            noisy_features_t = noisy_features.transpose(1, 2)

            # Cross-Modal Fusion (PPG内部的clean-noisy融合)
            for fusion_block in self.ppg_model.fusion_blocks:
                clean_features_t, noisy_features_t = fusion_block(clean_features_t, noisy_features_t)

            # 转回格式
            clean_features = clean_features_t.transpose(1, 2)
            noisy_features = noisy_features_t.transpose(1, 2)

            # 特征聚合
            combined = torch.cat([clean_features, noisy_features], dim=1)
            fused = self.ppg_model.feature_aggregation(combined)

            # 时序建模
            temporal = self.ppg_model.temporal_blocks(fused)

            # 特征细化
            refined = self.ppg_model.feature_refinement(temporal)

            # 自适应池化到n_epochs
            pooled = self.ppg_model.adaptive_pool(refined)  # (batch, d_model, n_epochs)

        # 转换为 (batch, n_epochs, d_model)
        features = pooled.transpose(1, 2)

        return features

    def forward(self, eeg_input: torch.Tensor,
                ppg_input: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            eeg_input: (batch, n_epochs, signal_length) = (batch, 6, 3000)
            ppg_input: (batch, signal_length) = (batch, 6144)
        Returns:
            output: (batch, n_epochs, n_classes) = (batch, 6, 4)
        """
        # 1. 提取特征 (冻结的Encoder)
        eeg_features = self.extract_eeg_features(eeg_input)  # (batch, 6, 256)
        ppg_features = self.extract_ppg_features(ppg_input)  # (batch, 6, 256)

        # 2. 双向Cross-Attention融合
        for fusion_block in self.fusion_blocks:
            eeg_features, ppg_features = fusion_block(eeg_features, ppg_features)

        # 3. 特征融合 (Concatenation)
        fused_features = torch.cat([eeg_features, ppg_features], dim=-1)  # (batch, 6, 512)
        fused_features = self.fusion_projection(fused_features)  # (batch, 6, 256)

        # 4. 分类
        output = self.classifier(fused_features)  # (batch, 6, 4)

        return output

    def get_trainable_params(self):
        """获取可训练参数数量"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        return {'total': total, 'trainable': trainable, 'frozen': frozen}


def load_pretrained_models(eeg_model_path: str, ppg_model_path: str, device: str = 'cuda'):
    """
    加载预训练的EEG和PPG模型

    Args:
        eeg_model_path: EEG模型权重路径
        ppg_model_path: PPG模型权重路径
        device: 设备
    Returns:
        eeg_model, ppg_model
    """
    # 导入模型类
    from short_window_eeg_model import ShortWindowAttnSleep
    from ppg_crossattn_shortwindow import PPGCrossAttnShortWindow

    # 创建模型实例
    eeg_model = ShortWindowAttnSleep(
        window_minutes=3,
        num_classes=4
    )

    ppg_model = PPGCrossAttnShortWindow(
        window_size='3min',  # 使用字符串格式
        n_classes=4
    )

    # 加载权重
    eeg_state = torch.load(eeg_model_path, map_location=device)
    ppg_state = torch.load(ppg_model_path, map_location=device)

    # 处理可能的state_dict包装
    if 'model_state_dict' in eeg_state:
        eeg_state = eeg_state['model_state_dict']
    if 'model_state_dict' in ppg_state:
        ppg_state = ppg_state['model_state_dict']

    eeg_model.load_state_dict(eeg_state)
    ppg_model.load_state_dict(ppg_state)

    eeg_model.to(device)
    ppg_model.to(device)

    eeg_model.eval()
    ppg_model.eval()

    print(f"Loaded EEG model from {eeg_model_path}")
    print(f"Loaded PPG model from {ppg_model_path}")

    return eeg_model, ppg_model


def create_fusion_model(eeg_model_path: str,
                        ppg_model_path: str,
                        device: str = 'cuda',
                        n_fusion_blocks: int = 2,
                        freeze_encoders: bool = True) -> EEGPPGCrossAttentionFusion:
    """
    创建Cross-Attention Fusion模型

    Args:
        eeg_model_path: EEG模型权重路径
        ppg_model_path: PPG模型权重路径
        device: 设备
        n_fusion_blocks: Cross-Attention块数量
        freeze_encoders: 是否冻结Encoder
    Returns:
        fusion_model
    """
    # 加载预训练模型
    eeg_model, ppg_model = load_pretrained_models(eeg_model_path, ppg_model_path, device)

    # 创建融合模型
    fusion_model = EEGPPGCrossAttentionFusion(
        eeg_model=eeg_model,
        ppg_model=ppg_model,
        d_model=256,
        n_heads=8,
        n_fusion_blocks=n_fusion_blocks,
        n_classes=4,
        dropout=0.1,
        freeze_encoders=freeze_encoders
    )

    fusion_model.to(device)

    # 打印参数统计
    params = fusion_model.get_trainable_params()
    print(f"\nModel parameters:")
    print(f"  Total: {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,}")
    print(f"  Frozen: {params['frozen']:,}")

    return fusion_model


def test_model():
    """测试模型结构"""
    print("Testing Cross-Attention Fusion Model...")

    # 模拟EEG和PPG模型（用于测试）
    class DummyEEGModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.mrcnn = nn.Linear(3000, 256)
            self.tce = nn.Identity()
            self.feature_compress = nn.Linear(256, 256)

        def forward(self, x):
            return self.feature_compress(self.mrcnn(x.squeeze(1)))

    class DummyPPGModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.clean_ppg_encoder = nn.Conv1d(1, 256, 7, stride=64, padding=3)
            self.noisy_ppg_encoder = nn.Conv1d(1, 256, 7, stride=64, padding=3)
            self.positional_encoding = nn.Parameter(torch.randn(1, 256, 1000))
            self.modality_weighting = lambda x, y: (torch.ones(x.size(0), 1, device=x.device) * 0.5,
                                                    torch.ones(x.size(0), 1, device=x.device) * 0.5)
            self.fusion_blocks = nn.ModuleList()
            self.feature_aggregation = nn.Conv1d(512, 256, 1)
            self.temporal_blocks = nn.Identity()
            self.feature_refinement = nn.Identity()
            self.adaptive_pool = nn.AdaptiveAvgPool1d(6)

        def add_noise_to_ppg(self, x):
            return x + torch.randn_like(x) * 0.1

    # 创建模拟模型
    eeg_model = DummyEEGModel()
    ppg_model = DummyPPGModel()

    # 创建融合模型
    fusion_model = EEGPPGCrossAttentionFusion(
        eeg_model=eeg_model,
        ppg_model=ppg_model,
        d_model=256,
        n_heads=8,
        n_fusion_blocks=2,
        n_classes=4,
        freeze_encoders=True
    )

    # 测试输入
    batch_size = 4
    eeg_input = torch.randn(batch_size, 6, 3000)  # (batch, n_epochs, signal_length)
    ppg_input = torch.randn(batch_size, 6144)  # (batch, signal_length)

    # 前向传播
    output = fusion_model(eeg_input, ppg_input)
    print(f"Output shape: {output.shape}")  # 应该是 (4, 6, 4)

    # 参数统计
    params = fusion_model.get_trainable_params()
    print(f"\nParameters:")
    print(f"  Total: {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,}")
    print(f"  Frozen: {params['frozen']:,}")

    print("\nTest passed!")


if __name__ == "__main__":
    test_model()