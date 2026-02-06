"""
EEG-PPG Cross-Attention Fusion Model with Mamba TCM
双向Cross-Attention融合 + Mamba时序建模

架构:
1. 冻结的EEG/PPG Encoder提取特征
2. Cross-Attention融合EEG和PPG特征
3. Mamba TCM进行跨epoch时序建模
4. 分类头输出预测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

# 尝试导入mamba
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba_ssm not installed. Using fallback GRU implementation.")


class MambaBlock(nn.Module):
    """Mamba块"""
    
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, 
                 expand: int = 2, dropout: float = 0.1):
        super().__init__()
        
        if MAMBA_AVAILABLE:
            self.mamba = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            )
        else:
            # Fallback: GRU
            self.mamba = nn.GRU(
                input_size=d_model,
                hidden_size=d_model,
                num_layers=1,
                batch_first=True
            )
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.is_mamba = MAMBA_AVAILABLE
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        if self.is_mamba:
            x = self.mamba(x)
        else:
            x, _ = self.mamba(x)
        
        x = self.dropout(x)
        x = self.norm(x + residual)
        return x


class MambaTCM(nn.Module):
    """Mamba-based Temporal Context Module (双向)"""
    
    def __init__(self, d_model: int, n_layers: int = 2, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2, dropout: float = 0.1):
        super().__init__()
        
        # 正向Mamba
        self.forward_layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand, dropout)
            for _ in range(n_layers)
        ])
        
        # 反向Mamba
        self.backward_layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand, dropout)
            for _ in range(n_layers)
        ])
        
        # 合并
        self.merge = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            output: (batch, seq_len, d_model)
        """
        # 正向
        forward_out = x
        for layer in self.forward_layers:
            forward_out = layer(forward_out)
        
        # 反向
        backward_out = torch.flip(x, dims=[1])
        for layer in self.backward_layers:
            backward_out = layer(backward_out)
        backward_out = torch.flip(backward_out, dims=[1])
        
        # 合并
        combined = torch.cat([forward_out, backward_out], dim=-1)
        output = self.merge(combined)
        
        return output


class MultiHeadCrossAttention(nn.Module):
    """多头交叉注意力"""
    
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
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
                value: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = query.shape
        residual = query
        
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        output = self.w_o(context)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        
        return output


class CrossModalFusionBlock(nn.Module):
    """双向交叉模态融合块"""
    
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.eeg_cross_attn = MultiHeadCrossAttention(d_model, n_heads, dropout)
        self.ppg_cross_attn = MultiHeadCrossAttention(d_model, n_heads, dropout)
        
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
        # EEG attend to PPG
        eeg_attended = self.eeg_cross_attn(eeg_features, ppg_features, ppg_features)
        # PPG attend to EEG
        ppg_attended = self.ppg_cross_attn(ppg_features, eeg_features, eeg_features)
        
        # FFN
        eeg_out = self.eeg_norm(eeg_attended + self.eeg_ffn(eeg_attended))
        ppg_out = self.ppg_norm(ppg_attended + self.ppg_ffn(ppg_attended))
        
        return eeg_out, ppg_out


class EEGPPGCrossAttentionMambaFusion(nn.Module):
    """
    EEG-PPG Cross-Attention Fusion with Mamba TCM
    
    结构:
    EEG Encoder (frozen) ─┐
                          ├─> Cross-Attention ─> Mamba TCM ─> Classifier
    PPG Encoder (frozen) ─┘
    """
    
    def __init__(self,
                 eeg_model: nn.Module,
                 ppg_model: nn.Module,
                 d_model: int = 256,
                 n_heads: int = 8,
                 n_fusion_blocks: int = 2,
                 n_mamba_layers: int = 2,
                 d_state: int = 16,
                 d_conv: int = 4,
                 expand: int = 2,
                 n_classes: int = 4,
                 dropout: float = 0.1,
                 freeze_encoders: bool = True):
        super().__init__()
        
        self.d_model = d_model
        self.n_classes = n_classes
        
        # Encoders
        self.eeg_model = eeg_model
        self.ppg_model = ppg_model
        
        if freeze_encoders:
            self._freeze_encoders()
        
        # Cross-Modal Fusion
        self.fusion_blocks = nn.ModuleList([
            CrossModalFusionBlock(d_model, n_heads, dropout)
            for _ in range(n_fusion_blocks)
        ])
        
        # 特征融合
        self.fusion_projection = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Mamba TCM
        self.mamba_tcm = MambaTCM(
            d_model=d_model,
            n_layers=n_mamba_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes)
        )
        
        self._init_weights()
    
    def _freeze_encoders(self):
        for param in self.eeg_model.parameters():
            param.requires_grad = False
        for param in self.ppg_model.parameters():
            param.requires_grad = False
        print("Encoders frozen.")
    
    def _init_weights(self):
        for module in [self.fusion_blocks, self.fusion_projection, 
                       self.mamba_tcm, self.classifier]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
    
    def extract_eeg_features(self, eeg_input: torch.Tensor) -> torch.Tensor:
        batch_size, num_epochs, signal_length = eeg_input.shape
        x_reshaped = eeg_input.view(batch_size * num_epochs, 1, signal_length)
        
        with torch.no_grad():
            epoch_features = self.eeg_model.mrcnn(x_reshaped)
            epoch_features = self.eeg_model.tce(epoch_features)
            epoch_features = epoch_features.contiguous().view(batch_size * num_epochs, -1)
            epoch_features = self.eeg_model.feature_compress(epoch_features)
        
        features = epoch_features.view(batch_size, num_epochs, -1)
        return features
    
    def extract_ppg_features(self, ppg_input: torch.Tensor) -> torch.Tensor:
        batch_size = ppg_input.size(0)
        ppg_input = ppg_input.unsqueeze(1)
        
        with torch.no_grad():
            noisy_ppg = self.ppg_model.add_noise_to_ppg(ppg_input)
            clean_features = self.ppg_model.clean_ppg_encoder(ppg_input)
            noisy_features = self.ppg_model.noisy_ppg_encoder(noisy_ppg)
            
            seq_len = clean_features.size(2)
            if seq_len <= self.ppg_model.positional_encoding.size(2):
                clean_features = clean_features + self.ppg_model.positional_encoding[:, :, :seq_len]
                noisy_features = noisy_features + self.ppg_model.positional_encoding[:, :, :seq_len]
            
            clean_weight, noisy_weight = self.ppg_model.modality_weighting(clean_features, noisy_features)
            clean_features = clean_features * clean_weight.unsqueeze(-1)
            noisy_features = noisy_features * noisy_weight.unsqueeze(-1)
            
            clean_features_t = clean_features.transpose(1, 2)
            noisy_features_t = noisy_features.transpose(1, 2)
            
            for fusion_block in self.ppg_model.fusion_blocks:
                clean_features_t, noisy_features_t = fusion_block(clean_features_t, noisy_features_t)
            
            clean_features = clean_features_t.transpose(1, 2)
            noisy_features = noisy_features_t.transpose(1, 2)
            
            combined = torch.cat([clean_features, noisy_features], dim=1)
            fused = self.ppg_model.feature_aggregation(combined)
            temporal = self.ppg_model.temporal_blocks(fused)
            refined = self.ppg_model.feature_refinement(temporal)
            pooled = self.ppg_model.adaptive_pool(refined)
        
        features = pooled.transpose(1, 2)
        return features
    
    def forward(self, eeg_input: torch.Tensor, ppg_input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            eeg_input: (batch, 6, 3000)
            ppg_input: (batch, 6144)
        Returns:
            output: (batch, 6, 4)
        """
        # 1. 提取特征
        eeg_features = self.extract_eeg_features(eeg_input)  # (batch, 6, 256)
        ppg_features = self.extract_ppg_features(ppg_input)  # (batch, 6, 256)
        
        # 2. Cross-Attention融合
        for fusion_block in self.fusion_blocks:
            eeg_features, ppg_features = fusion_block(eeg_features, ppg_features)
        
        # 3. 特征拼接和投影
        fused = torch.cat([eeg_features, ppg_features], dim=-1)  # (batch, 6, 512)
        fused = self.fusion_projection(fused)  # (batch, 6, 256)
        
        # 4. Mamba TCM时序建模
        temporal = self.mamba_tcm(fused)  # (batch, 6, 256)
        
        # 5. 分类
        output = self.classifier(temporal)  # (batch, 6, 4)
        
        return output
    
    def get_trainable_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        return {'total': total, 'trainable': trainable, 'frozen': frozen}


def create_mamba_fusion_model(eeg_model_path: str,
                               ppg_model_path: str,
                               device: str = 'cuda',
                               n_fusion_blocks: int = 2,
                               n_mamba_layers: int = 2,
                               freeze_encoders: bool = True):
    """创建带Mamba TCM的融合模型"""
    
    from short_window_eeg_model import ShortWindowAttnSleep
    from ppg_crossattn_shortwindow import PPGCrossAttnShortWindow
    
    # 加载EEG模型
    eeg_model = ShortWindowAttnSleep(window_minutes=3, num_classes=4)
    eeg_state = torch.load(eeg_model_path, map_location=device)
    if 'model_state_dict' in eeg_state:
        eeg_state = eeg_state['model_state_dict']
    eeg_model.load_state_dict(eeg_state)
    eeg_model.to(device)
    eeg_model.eval()
    
    # 加载PPG模型
    ppg_model = PPGCrossAttnShortWindow(window_size='3min', n_classes=4)
    ppg_state = torch.load(ppg_model_path, map_location=device)
    if 'model_state_dict' in ppg_state:
        ppg_state = ppg_state['model_state_dict']
    ppg_model.load_state_dict(ppg_state)
    ppg_model.to(device)
    ppg_model.eval()
    
    print(f"Loaded EEG model from {eeg_model_path}")
    print(f"Loaded PPG model from {ppg_model_path}")
    
    # 创建融合模型
    model = EEGPPGCrossAttentionMambaFusion(
        eeg_model=eeg_model,
        ppg_model=ppg_model,
        d_model=256,
        n_heads=8,
        n_fusion_blocks=n_fusion_blocks,
        n_mamba_layers=n_mamba_layers,
        d_state=16,
        d_conv=4,
        expand=2,
        n_classes=4,
        dropout=0.1,
        freeze_encoders=freeze_encoders
    )
    
    model.to(device)
    
    params = model.get_trainable_params()
    print(f"\nModel parameters:")
    print(f"  Total: {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,}")
    print(f"  Frozen: {params['frozen']:,}")
    print(f"  Mamba available: {MAMBA_AVAILABLE}")
    
    return model
