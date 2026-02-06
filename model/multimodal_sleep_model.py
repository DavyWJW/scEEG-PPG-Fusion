import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm



class ResConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(out_channels)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        if in_channels != out_channels:
            self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = None

    def forward(self, x):
        residual = x

        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))

        x = self.pool(x)

        if self.residual_conv is not None:
            residual = self.residual_conv(residual)

        residual = F.max_pool1d(residual, kernel_size=2, stride=2)

        return x + residual


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.LeakyReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=7, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = 6
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels
            out_channels = num_channels
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)



class CrossModalAttention(nn.Module):
    def __init__(self, feature_dim, num_heads=8):
        super(CrossModalAttention, self).__init__()
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        self.head_dim = feature_dim // num_heads

        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)

        self.out_proj = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, ppg_features, ecg_features):
        batch_size, channels, length = ppg_features.shape


        ppg_features = ppg_features.transpose(1, 2)
        ecg_features = ecg_features.transpose(1, 2)


        Q = self.query_proj(ppg_features)
        K = self.key_proj(ecg_features)
        V = self.value_proj(ecg_features)


        Q = Q.view(batch_size, length, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, length, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, length, self.num_heads, self.head_dim).transpose(1, 2)


        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)


        context = torch.matmul(attention_probs, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, length, self.feature_dim)


        output = self.out_proj(context)


        output = output + ppg_features


        output = output.transpose(1, 2)

        return output



class MultiModalSleepNet(nn.Module):
    def __init__(self, fusion_strategy='attention'):
        super(MultiModalSleepNet, self).__init__()

        self.fusion_strategy = fusion_strategy


        self.ppg_encoder = nn.Sequential(
            ResConvBlock(1, 16),
            ResConvBlock(16, 16),
            ResConvBlock(16, 32),
            ResConvBlock(32, 32),
            ResConvBlock(32, 64),
            ResConvBlock(64, 64),
            ResConvBlock(64, 128),
            ResConvBlock(128, 256),
        )


        self.ecg_encoder = nn.Sequential(
            ResConvBlock(1, 16),
            ResConvBlock(16, 16),
            ResConvBlock(16, 32),
            ResConvBlock(32, 32),
            ResConvBlock(32, 64),
            ResConvBlock(64, 64),
            ResConvBlock(64, 128),
            ResConvBlock(128, 256),
        )


        if fusion_strategy == 'concat':
            self.fusion_dim = 512  # 256 + 256
            self.fusion_layer = nn.Identity()
        elif fusion_strategy == 'attention':
            self.fusion_dim = 256
            self.fusion_layer = CrossModalAttention(256)
        elif fusion_strategy == 'gated':
            self.fusion_dim = 256
            self.ppg_gate = nn.Sequential(
                nn.Linear(256, 256),
                nn.Sigmoid()
            )
            self.ecg_gate = nn.Sequential(
                nn.Linear(256, 256),
                nn.Sigmoid()
            )


        self.dense = nn.Linear(self.fusion_dim * 4, 128)
        self.tcnblock1 = TemporalConvNet(128, 128, kernel_size=7, dropout=0.2)
        self.tcnblock2 = TemporalConvNet(128, 128, kernel_size=7, dropout=0.2)


        self.final_conv = nn.Conv1d(128, 4, 1)

    def forward(self, ppg, ecg):

        ppg_features = self.ppg_encoder(ppg)  # (B, 256, 4800)
        ecg_features = self.ecg_encoder(ecg)  # (B, 256, 4800)


        if self.fusion_strategy == 'concat':
            fused_features = torch.cat([ppg_features, ecg_features], dim=1)
        elif self.fusion_strategy == 'attention':
            fused_features = self.fusion_layer(ppg_features, ecg_features)
        elif self.fusion_strategy == 'gated':

            ppg_gate = self.ppg_gate(ppg_features.mean(dim=2)).unsqueeze(2)
            ecg_gate = self.ecg_gate(ecg_features.mean(dim=2)).unsqueeze(2)
            fused_features = ppg_gate * ppg_features + ecg_gate * ecg_features


        batch_size, channels, length = fused_features.shape


        fused_features = fused_features.view(batch_size, channels, 1200, 4)
        fused_features = fused_features.permute(0, 1, 3, 2).contiguous()
        fused_features = fused_features.view(batch_size, -1, 1200)


        fused_features = fused_features.transpose(1, 2)  # (B, 1200, channels*4)
        fused_features = self.dense(fused_features)  # (B, 1200, 128)
        fused_features = fused_features.transpose(1, 2)  # (B, 128, 1200)


        x = self.tcnblock1(fused_features)
        x = self.tcnblock2(x)


        x = self.final_conv(x)  # (B, 4, 1200)
        x = F.softmax(x, dim=1)

        return x



class SleepPPGNet(nn.Module):
    def __init__(self):
        super(SleepPPGNet, self).__init__()

        self.resconv_blocks = nn.Sequential(
            ResConvBlock(1, 16),
            ResConvBlock(16, 16),
            ResConvBlock(16, 32),
            ResConvBlock(32, 32),
            ResConvBlock(32, 64),
            ResConvBlock(64, 64),
            ResConvBlock(64, 128),
            ResConvBlock(128, 256),
        )

        self.dense = nn.Linear(1024, 128)
        self.tcnblock1 = TemporalConvNet(128, 128, kernel_size=7, dropout=0.2)
        self.tcnblock2 = TemporalConvNet(128, 128, kernel_size=7, dropout=0.2)
        self.final_conv = nn.Conv1d(128, 4, 1)

    def forward(self, x):
        x = self.resconv_blocks(x)

        batch_size, channels, length = x.shape
        x = x.view(batch_size, channels, 1200, length // 1200)
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(batch_size, -1, 1200)

        x = x.transpose(1, 2)
        x = self.dense(x)
        x = x.transpose(1, 2)

        x = self.tcnblock1(x)
        x = self.tcnblock2(x)

        x = self.final_conv(x)
        x = F.softmax(x, dim=1)

        return x


def test_models():

    batch_size = 2
    ppg = torch.randn(batch_size, 1, 1228800)
    ecg = torch.randn(batch_size, 1, 1228800)


    print("Testing PPG-only model...")
    ppg_model = SleepPPGNet()
    ppg_output = ppg_model(ppg)
    print(f"PPG-only output shape: {ppg_output.shape}")


    print("\nTesting MultiModal model...")
    mm_model = MultiModalSleepNet(fusion_strategy='attention')
    mm_output = mm_model(ppg, ecg)
    print(f"MultiModal output shape: {mm_output.shape}")


    ppg_params = sum(p.numel() for p in ppg_model.parameters())
    mm_params = sum(p.numel() for p in mm_model.parameters())
    print(f"\nPPG-only parameters: {ppg_params:,}")
    print(f"MultiModal parameters: {mm_params:,}")


if __name__ == "__main__":
    test_models()