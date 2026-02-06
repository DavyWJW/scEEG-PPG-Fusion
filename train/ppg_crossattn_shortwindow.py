"""
PPG + Unfiltered PPG Cross-Attention Model - Short Window Version

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Dict, Tuple


class ResConvBlock(nn.Module):

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


    WINDOW_CONFIGS = {
        '30s':   (1024,      1,    4),   # 30s = 1 epoch
        '1min':  (2048,      2,    5),   # 1min = 2 epochs
        '3min':  (6144,      6,    6),
        '5min':  (10240,     10,   6),
        '10min': (20480,     20,   7),
        '30min': (61440,     60,   8),
        '10H':   (1228800,   1200, 9),
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

        super().__init__()
        
        self.n_classes = n_classes
        self.d_model = d_model
        self.window_size = window_size
        self.min_feature_length = min_feature_length
        self.deterministic_noise = deterministic_noise
        

        if window_size not in self.WINDOW_CONFIGS:
            raise ValueError(f"Unsupported window_size: {window_size}. "
                           f"Choose from {list(self.WINDOW_CONFIGS.keys())}")
        
        self.input_samples, self.n_epochs, self.encoder_layers = self.WINDOW_CONFIGS[window_size]
        

        self.noise_config = noise_config or {
            'noise_level': 0.1,
            'drift_amplitude': 0.1,
            'drift_frequency': 0.1,
            'spike_probability': 0.01,
            'spike_amplitude': 0.5
        }
        

        self.encoder_downsample = 2 ** self.encoder_layers
        self.expected_feature_length = self.input_samples // self.encoder_downsample
        
        print(f"[PPGCrossAttnShortWindow] Window: {window_size}")
        print(f"  Input samples: {self.input_samples}")
        print(f"  Output epochs: {self.n_epochs}")
        print(f"  Encoder layers: {self.encoder_layers} (downsample: {self.encoder_downsample}x)")
        print(f"  Expected feature length: {self.expected_feature_length}")
        

        self.clean_ppg_encoder = self._create_encoder(d_model, self.encoder_layers)
        self.noisy_ppg_encoder = self._create_encoder(d_model, self.encoder_layers)
        

        max_pos_len = max(3000, self.expected_feature_length + 100)
        self.positional_encoding = self._create_positional_encoding(d_model, max_pos_len)
        

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
        

        self.adaptive_pool = nn.AdaptiveAvgPool1d(self.n_epochs)
        

        self.classifier = nn.Sequential(
            nn.Conv1d(d_model, 128, 1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(128, n_classes, 1)
        )
        

        self._init_weights()
    
    def _create_encoder(self, d_model: int, n_layers: int) -> nn.Sequential:

        layers = []
        

        if n_layers <= 4:
            channels = [1, 32, 64, 128, d_model][:n_layers + 1]
        elif n_layers <= 6:
            channels = [1, 16, 32, 64, 128, 256, d_model][:n_layers + 1]
        else:
            channels = [1, 16, 32, 64, 128, 256, 256, 256, 256, d_model][:n_layers + 1]
        

        channels[-1] = d_model
        
        for i in range(len(channels) - 1):
            layers.append(ResConvBlock(channels[i], channels[i + 1], stride=2))
        
        return nn.Sequential(*layers)
    
    def _create_positional_encoding(self, d_model: int, max_len: int) -> nn.Parameter:

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return nn.Parameter(pe.unsqueeze(0).transpose(1, 2), requires_grad=False)
    
    def _create_temporal_blocks(self, d_model: int) -> nn.Sequential:

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

        batch_size, _, length = clean_ppg.shape
        device = clean_ppg.device
        

        if self.deterministic_noise or seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(seed if seed is not None else 42)
        else:
            generator = None
        
        noisy_ppg = clean_ppg.clone()
        

        if generator:
            gaussian_noise = torch.randn(batch_size, 1, length, device=device, 
                                        generator=generator) * self.noise_config['noise_level']
        else:
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
        
        if generator:
            spike_mask = torch.rand(batch_size, 1, length, device=device, 
                                   generator=generator) < spike_prob
            spike_values = torch.randn(batch_size, 1, length, device=device,
                                      generator=generator) * spike_amp
        else:
            spike_mask = torch.rand(batch_size, 1, length, device=device) < spike_prob
            spike_values = torch.randn(batch_size, 1, length, device=device) * spike_amp
        
        spikes = spike_mask.float() * spike_values
        

        kernel_size = 5
        padding = kernel_size // 2
        smoothing_kernel = torch.ones(1, 1, kernel_size, device=device) / kernel_size
        spikes = F.conv1d(spikes, smoothing_kernel, padding=padding)
        noisy_ppg = noisy_ppg + spikes
        

        if generator:
            emg_noise = torch.randn(batch_size, 1, length, device=device,
                                   generator=generator) * 0.05
        else:
            emg_noise = torch.randn_like(clean_ppg) * 0.05
        noisy_ppg = noisy_ppg + emg_noise
        
        return noisy_ppg
    
    def forward(self, ppg: torch.Tensor) -> torch.Tensor:

        if ppg.dim() == 2:
            ppg = ppg.unsqueeze(1)
        
        batch_size = ppg.size(0)
        input_length = ppg.size(2)
        

        if input_length != self.input_samples:

            if abs(input_length - self.input_samples) > self.input_samples * 0.1:
                raise ValueError(f"Input length {input_length} doesn't match expected "
                               f"{self.input_samples} for window size {self.window_size}")
        

        ppg_unfiltered = self.add_noise_to_ppg(ppg)
        

        clean_features = self.clean_ppg_encoder(ppg)
        noisy_features = self.noisy_ppg_encoder(ppg_unfiltered)
        

        seq_len = clean_features.size(2)
        if seq_len <= self.positional_encoding.size(2):
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

        output_features = self.adaptive_pool(refined_features)
        

        logits = self.classifier(output_features)
        
        return logits
    
    def predict(self, ppg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        logits = self.forward(ppg)
        probabilities = F.softmax(logits, dim=1)
        predictions = logits.argmax(dim=1)
        return predictions, probabilities
    
    def get_config(self) -> Dict:

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

    return PPGCrossAttnShortWindow(window_size=window_size, **kwargs)


def test_all_window_sizes():

    print("=" * 80)
    print("Testing PPG Cross-Attention Model for All Window Sizes")
    print("=" * 80)
    
    results = {}
    
    for window_size in PPGCrossAttnShortWindow.WINDOW_CONFIGS.keys():
        print(f"\n{'='*60}")
        print(f"Testing window size: {window_size}")
        print("=" * 60)
        
        try:

            model = create_model_for_window(window_size)
            

            config = model.get_config()
            input_samples = config['input_samples']
            n_epochs = config['n_epochs']
            

            batch_size = 2
            ppg = torch.randn(batch_size, 1, input_samples)
            

            with torch.no_grad():
                logits = model(ppg)
            
            print(f"Input shape: {ppg.shape}")
            print(f"Output shape: {logits.shape}")
            print(f"Expected: (B={batch_size}, C=4, T={n_epochs})")
            

            expected_shape = (batch_size, 4, n_epochs)
            assert logits.shape == expected_shape, \
                f"Shape mismatch: got {logits.shape}, expected {expected_shape}"
            

            print(f"Output range: [{logits.min():.4f}, {logits.max():.4f}]")
            

            predictions, probs = model.predict(ppg)
            print(f"Predictions shape: {predictions.shape}")
            print(f"Probabilities shape: {probs.shape}")
            print(f"Prob sum per position (should be 1.0): {probs.sum(dim=1).mean():.6f}")
            

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
    test_all_window_sizes()

    benchmark_inference_time()
