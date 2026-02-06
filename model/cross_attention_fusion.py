"""
EEG-PPG Cross-Attention Fusion Model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class MultiHeadCrossAttention(nn.Module):


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


        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)


        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)


        context = torch.matmul(attention_weights, V)


        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)


        output = self.w_o(context)
        output = self.dropout(output)


        output = self.layer_norm(output + residual)

        return output, attention_weights


class CrossModalFusionBlock(nn.Module):


    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        # EEG attend to PPG
        self.eeg_cross_attn = MultiHeadCrossAttention(d_model, n_heads, dropout)

        # PPG attend to EEG
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
        """
        Args:
            eeg_features: (batch, seq_len, d_model)
            ppg_features: (batch, seq_len, d_model)
        Returns:
            eeg_out: (batch, seq_len, d_model)
            ppg_out: (batch, seq_len, d_model)
        """
        # EEG attend to PPG
        eeg_attended, _ = self.eeg_cross_attn(eeg_features, ppg_features, ppg_features)

        # PPG attend to EEG
        ppg_attended, _ = self.ppg_cross_attn(ppg_features, eeg_features, eeg_features)

        # FFN
        eeg_out = self.eeg_norm(eeg_attended + self.eeg_ffn(eeg_attended))
        ppg_out = self.ppg_norm(ppg_attended + self.ppg_ffn(ppg_attended))

        return eeg_out, ppg_out


class EEGPPGCrossAttentionFusion(nn.Module):
    """
    EEG-PPG Cross-Attention Fusion Model

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


        self.eeg_model = eeg_model
        self.ppg_model = ppg_model


        if freeze_encoders:
            self._freeze_encoders()


        self.fusion_blocks = nn.ModuleList([
            CrossModalFusionBlock(d_model, n_heads, dropout)
            for _ in range(n_fusion_blocks)
        ])


        self.fusion_projection = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )


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

        print("Encoders frozen. Only fusion layers and classifier will be trained.")

    def _init_weights(self):

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
        Args:
            eeg_input: (batch, n_epochs, signal_length) = (batch, 6, 3000)
        Returns:
            features: (batch, n_epochs, d_model) = (batch, 6, 256)
        """
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
        """
        Args:
            ppg_input: (batch, signal_length) = (batch, 6144) for 3min
        Returns:
            features: (batch, n_epochs, d_model) = (batch, 6, 256)
        """
        batch_size = ppg_input.size(0)


        if ppg_input.dim() == 2:
            ppg_input = ppg_input.unsqueeze(1)  # (batch, 1, signal_length)

        with torch.no_grad():

            ppg_noisy = self.ppg_model.add_noise_to_ppg(ppg_input)


            clean_features = self.ppg_model.clean_ppg_encoder(ppg_input)
            noisy_features = self.ppg_model.noisy_ppg_encoder(ppg_noisy)


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


            pooled = self.ppg_model.adaptive_pool(refined)  # (batch, d_model, n_epochs)


        features = pooled.transpose(1, 2)

        return features

    def forward(self, eeg_input: torch.Tensor,
                ppg_input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            eeg_input: (batch, n_epochs, signal_length) = (batch, 6, 3000)
            ppg_input: (batch, signal_length) = (batch, 6144)
        Returns:
            output: (batch, n_epochs, n_classes) = (batch, 6, 4)
        """

        eeg_features = self.extract_eeg_features(eeg_input)  # (batch, 6, 256)
        ppg_features = self.extract_ppg_features(ppg_input)  # (batch, 6, 256)


        for fusion_block in self.fusion_blocks:
            eeg_features, ppg_features = fusion_block(eeg_features, ppg_features)


        fused_features = torch.cat([eeg_features, ppg_features], dim=-1)  # (batch, 6, 512)
        fused_features = self.fusion_projection(fused_features)  # (batch, 6, 256)


        output = self.classifier(fused_features)  # (batch, 6, 4)

        return output

    def get_trainable_params(self):

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        return {'total': total, 'trainable': trainable, 'frozen': frozen}


def load_pretrained_models(eeg_model_path: str, ppg_model_path: str, device: str = 'cuda'):

    from short_window_eeg_model import ShortWindowAttnSleep
    from ppg_crossattn_shortwindow import PPGCrossAttnShortWindow


    eeg_model = ShortWindowAttnSleep(
        window_minutes=3,
        num_classes=4
    )

    ppg_model = PPGCrossAttnShortWindow(
        window_size='3min',
        n_classes=4
    )


    eeg_state = torch.load(eeg_model_path, map_location=device)
    ppg_state = torch.load(ppg_model_path, map_location=device)


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

    eeg_model, ppg_model = load_pretrained_models(eeg_model_path, ppg_model_path, device)


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


    params = fusion_model.get_trainable_params()
    print(f"\nModel parameters:")
    print(f"  Total: {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,}")
    print(f"  Frozen: {params['frozen']:,}")

    return fusion_model


def test_model():

    print("Testing Cross-Attention Fusion Model...")


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


    eeg_model = DummyEEGModel()
    ppg_model = DummyPPGModel()


    fusion_model = EEGPPGCrossAttentionFusion(
        eeg_model=eeg_model,
        ppg_model=ppg_model,
        d_model=256,
        n_heads=8,
        n_fusion_blocks=2,
        n_classes=4,
        freeze_encoders=True
    )


    batch_size = 4
    eeg_input = torch.randn(batch_size, 6, 3000)  # (batch, n_epochs, signal_length)
    ppg_input = torch.randn(batch_size, 6144)  # (batch, signal_length)


    output = fusion_model(eeg_input, ppg_input)
    print(f"Output shape: {output.shape}")  # 应该是 (4, 6, 4)


    params = fusion_model.get_trainable_params()
    print(f"\nParameters:")
    print(f"  Total: {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,}")
    print(f"  Frozen: {params['frozen']:,}")

    print("\nTest passed!")


if __name__ == "__main__":
    test_model()