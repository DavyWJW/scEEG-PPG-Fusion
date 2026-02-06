"""
PPG + Unfiltered PPG Cross-Attention Model

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import signal
import math


class PPGUnfilteredCrossAttention(nn.Module):


    def __init__(self, n_classes=4, d_model=256, n_heads=8, n_fusion_blocks=3,
                 noise_config=None):
        super().__init__()


        self.noise_config = noise_config or {
            'noise_level': 0.1,
            'drift_amplitude': 0.1,
            'drift_frequency': 0.1,
            'spike_probability': 0.01,
            'spike_amplitude': 0.5
        }


        self.clean_ppg_encoder = self._create_encoder(d_model)
        self.noisy_ppg_encoder = self._create_encoder(d_model)


        self.positional_encoding = self._create_positional_encoding(d_model, 3000)


        self.fusion_blocks = nn.ModuleList([
            CrossModalFusionBlock(d_model, n_heads)
            for _ in range(n_fusion_blocks)
        ])


        self.modality_weighting = AdaptiveModalityWeighting(d_model)


        self.feature_aggregation = nn.Sequential(
            nn.Conv1d(d_model * 2, d_model, 1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(0.2)
        )


        self.temporal_blocks = self._create_temporal_blocks(d_model)


        self.feature_refinement = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )


        self.classifier = nn.Sequential(
            nn.Conv1d(d_model, 128, 1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(128, n_classes, 1)
        )

    def _create_encoder(self, d_model):

        from multimodal_model_crossattn import ResConvBlock

        return nn.Sequential(
            ResConvBlock(1, 16, stride=2),  # 1228800 -> 614400
            ResConvBlock(16, 32, stride=2),  # 614400 -> 307200
            ResConvBlock(32, 64, stride=2),  # 307200 -> 153600
            ResConvBlock(64, 128, stride=2),  # 153600 -> 76800
            ResConvBlock(128, 256, stride=2),  # 76800 -> 38400
            ResConvBlock(256, 256, stride=2),  # 38400 -> 19200
            ResConvBlock(256, 256, stride=2),  # 19200 -> 9600
            ResConvBlock(256, 256, stride=2),  # 9600 -> 4800
            ResConvBlock(256, d_model, stride=2)  # 4800 -> 2400
        )

    def _create_positional_encoding(self, d_model, max_len):

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return nn.Parameter(pe.unsqueeze(0).transpose(1, 2), requires_grad=False)

    def _create_temporal_blocks(self, d_model):

        from multimodal_model_crossattn import TemporalConvBlock

        return nn.Sequential(
            TemporalConvBlock(d_model, d_model, kernel_size=7, dilation=1),
            TemporalConvBlock(d_model, d_model, kernel_size=7, dilation=2),
            TemporalConvBlock(d_model, d_model, kernel_size=7, dilation=4),
            TemporalConvBlock(d_model, d_model, kernel_size=7, dilation=8)
        )

    def add_noise_to_ppg(self, clean_ppg):

        batch_size, _, length = clean_ppg.shape
        device = clean_ppg.device


        noisy_ppg = clean_ppg.clone()


        gaussian_noise = torch.randn_like(clean_ppg) * self.noise_config['noise_level']
        noisy_ppg = noisy_ppg + gaussian_noise


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


        spike_prob = self.noise_config['spike_probability']
        spike_amp = self.noise_config['spike_amplitude']


        spike_mask = torch.rand(batch_size, 1, length, device=device) < spike_prob
        spike_values = torch.randn(batch_size, 1, length, device=device) * spike_amp
        spikes = spike_mask.float() * spike_values


        kernel_size = 5
        padding = kernel_size // 2
        smoothing_kernel = torch.ones(1, 1, kernel_size, device=device) / kernel_size
        spikes = F.conv1d(spikes, smoothing_kernel, padding=padding)

        noisy_ppg = noisy_ppg + spikes


        emg_noise = torch.randn_like(clean_ppg) * 0.05

        noisy_ppg = noisy_ppg + emg_noise

        return noisy_ppg

    def forward(self, ppg):
        batch_size = ppg.size(0)


        ppg_unfiltered = self.add_noise_to_ppg(ppg)


        clean_features = self.clean_ppg_encoder(ppg)  # (B, d_model, 2400)
        noisy_features = self.noisy_ppg_encoder(ppg_unfiltered)  # (B, d_model, 2400)


        seq_len = clean_features.size(2)
        clean_features = clean_features + self.positional_encoding[:, :, :seq_len]
        noisy_features = noisy_features + self.positional_encoding[:, :, :seq_len]


        clean_weight, noisy_weight = self.modality_weighting(clean_features, noisy_features)


        clean_features_weighted = clean_features * clean_weight.unsqueeze(-1)
        noisy_features_weighted = noisy_features * noisy_weight.unsqueeze(-1)


        clean_features_t = clean_features_weighted.transpose(1, 2)
        noisy_features_t = noisy_features_weighted.transpose(1, 2)


        for fusion_block in self.fusion_blocks:
            clean_features_t, noisy_features_t = fusion_block(clean_features_t, noisy_features_t)


        clean_features = clean_features_t.transpose(1, 2)
        noisy_features = noisy_features_t.transpose(1, 2)


        combined_features = torch.cat([clean_features, noisy_features], dim=1)
        fused_features = self.feature_aggregation(combined_features)


        temporal_features = self.temporal_blocks(fused_features)


        refined_features = self.feature_refinement(temporal_features)


        output_features = F.avg_pool1d(refined_features, kernel_size=2, stride=2)


        if output_features.size(2) != 1200:
            output_features = F.interpolate(output_features, size=1200, mode='linear', align_corners=False)


        output = self.classifier(output_features)  # (B, 4, 1200)
        output = F.softmax(output, dim=1)

        return output

    def get_modality_weights(self):

        if hasattr(self, 'clean_weight') and hasattr(self, 'noisy_weight'):
            return self.clean_weight, self.noisy_weight
        else:
            return None, None


class CrossModalFusionBlock(nn.Module):


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


class MultiHeadCrossAttention(nn.Module):


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


        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )


        output = self.w_o(context)
        output = self.dropout(output)


        output = self.layer_norm(output + query)

        return output, attention_weights


class AdaptiveModalityWeighting(nn.Module):


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


def test_model():

    print("Testing PPG + Unfiltered PPG Cross-Attention Model...")


    model = PPGUnfilteredCrossAttention()


    ppg = torch.randn(2, 1, 1228800)


    output = model(ppg)
    print(f"Output shape: {output.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")


    clean_ppg = torch.randn(1, 1, 1000)
    noisy_ppg = model.add_noise_to_ppg(clean_ppg)

    print(f"\nNoise statistics:")
    print(f"Clean PPG - mean: {clean_ppg.mean():.4f}, std: {clean_ppg.std():.4f}")
    print(f"Noisy PPG - mean: {noisy_ppg.mean():.4f}, std: {noisy_ppg.std():.4f}")
    print(f"Noise level: {(noisy_ppg - clean_ppg).std():.4f}")


    if torch.cuda.is_available():
        model = model.cuda()
        ppg = ppg.cuda()

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        with torch.no_grad():
            output = model(ppg)

        torch.cuda.synchronize()
        max_memory = torch.cuda.max_memory_allocated() / 1024 ** 3
        print(f"\nPeak GPU memory usage: {max_memory:.2f} GB")


if __name__ == "__main__":
    test_model()