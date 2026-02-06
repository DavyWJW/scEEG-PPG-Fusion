"""
短窗口EEG AttnSleep模型
处理多个连续的30秒EEG epochs（3分钟、5分钟、10分钟、30分钟）
对每个epoch进行独立特征提取，然后通过序列建模整合时序信息
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from copy import deepcopy


# ============================================================================
# 基础组件（来自原始AttnSleep）
# ============================================================================

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, 1)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return torch.nn.functional.gelu(x)


class MRCNN(nn.Module):
    """多分辨率CNN特征提取器"""

    def __init__(self, afr_reduced_cnn_size):
        super(MRCNN, self).__init__()
        drate = 0.5
        self.GELU = GELU()

        self.features1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=50, stride=6, bias=False, padding=24),
            nn.BatchNorm1d(64),
            self.GELU,
            nn.MaxPool1d(kernel_size=8, stride=2, padding=4),
            nn.Dropout(drate),
            nn.Conv1d(64, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            self.GELU,
            nn.Conv1d(128, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            self.GELU,
            nn.MaxPool1d(kernel_size=4, stride=4, padding=2)
        )

        self.features2 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=400, stride=50, bias=False, padding=200),
            nn.BatchNorm1d(64),
            self.GELU,
            nn.MaxPool1d(kernel_size=4, stride=2, padding=2),
            nn.Dropout(drate),
            nn.Conv1d(64, 128, kernel_size=7, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(128),
            self.GELU,
            nn.Conv1d(128, 128, kernel_size=7, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(128),
            self.GELU,
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.dropout = nn.Dropout(drate)
        self.inplanes = 128
        self.AFR = self._make_layer(SEBasicBlock, afr_reduced_cnn_size, 1)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.features1(x)
        x2 = self.features2(x)
        x_concat = torch.cat((x1, x2), dim=2)
        x_concat = self.dropout(x_concat)
        x_concat = self.AFR(x_concat)
        return x_concat


# ============================================================================
# 序列建模组件
# ============================================================================

class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]


# ============================================================================
# AttnSleep原始Transformer组件
# ============================================================================

def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([deepcopy(module) for _ in range(N)])


def attention(query, key, value, dropout=None):
    """Implementation of Scaled dot product attention"""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        self.__padding = (kernel_size - 1) * dilation
        super(CausalConv1d, self).__init__(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride,
            padding=self.__padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        if self.__padding != 0:
            return result[:, :, :-self.__padding]
        return result


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, afr_reduced_cnn_size, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.convs = clones(CausalConv1d(afr_reduced_cnn_size, afr_reduced_cnn_size, kernel_size=7, stride=1), 3)
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value):
        nbatches = query.size(0)
        query = query.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        key = self.convs[1](key).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        value = self.convs[2](value).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        x, self.attn = attention(query, key, value, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linear(x)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerOutput(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerOutput, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, afr_reduced_cnn_size, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer_output = clones(SublayerOutput(size, dropout), 2)
        self.size = size
        self.conv = CausalConv1d(afr_reduced_cnn_size, afr_reduced_cnn_size, kernel_size=7, stride=1, dilation=1)

    def forward(self, x_in):
        query = self.conv(x_in)
        x = self.sublayer_output[0](query, lambda x: self.self_attn(query, x_in, x_in))
        return self.sublayer_output[1](x, self.feed_forward)


class TCE(nn.Module):
    """Transformer Encoder - Stack of N layers"""

    def __init__(self, layer, N):
        super(TCE, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class TemporalTransformer(nn.Module):
    """时序Transformer用于epoch间建模"""

    def __init__(self, d_model, n_heads=4, n_layers=2, dropout=0.1):
        super(TemporalTransformer, self).__init__()

        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        return x


# ============================================================================
# 短窗口AttnSleep模型
# ============================================================================

class ShortWindowAttnSleep(nn.Module):
    """
    短窗口AttnSleep模型
    处理多个连续的30秒EEG epochs
    """

    def __init__(self, window_minutes=5, num_classes=4):
        super(ShortWindowAttnSleep, self).__init__()

        self.window_minutes = window_minutes
        self.epochs_per_window = window_minutes * 2  # 每分钟2个30秒epoch
        self.num_classes = num_classes

        print(f"\nShortWindowAttnSleep Configuration:")
        print(f"  Window length: {window_minutes} minutes")
        print(f"  Epochs per window: {self.epochs_per_window}")
        print(f"  Number of classes: {num_classes}")

        # AttnSleep参数
        afr_reduced_cnn_size = 30

        # 单epoch特征提取器（共享权重）
        self.mrcnn = MRCNN(afr_reduced_cnn_size)

        # 动态计算d_model（MRCNN输出的时间维度）
        # 测试一下实际输出维度
        with torch.no_grad():
            test_input = torch.randn(1, 1, 3000)  # 假设30秒@100Hz = 3000采样点
            test_output = self.mrcnn(test_input)
            d_model = test_output.shape[2]  # 时间维度
            print(f"  MRCNN output shape: {test_output.shape}")
            print(f"  d_model (temporal dim): {d_model}")

        # Epoch内注意力（原AttnSleep的TCE）
        h = 7
        # 确保d_model能被h整除
        if d_model % h != 0:
            # 调整h使其能整除d_model
            for new_h in [8, 5, 4, 2, 1]:
                if d_model % new_h == 0:
                    h = new_h
                    break
            print(f"  Adjusted number of heads to: {h}")

        d_ff = 120
        dropout = 0.1
        N = 2

        attn = MultiHeadedAttention(h, d_model, afr_reduced_cnn_size)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.tce = TCE(EncoderLayer(d_model, deepcopy(attn), deepcopy(ff), afr_reduced_cnn_size, dropout), N)

        # Epoch特征维度
        self.epoch_feature_dim = d_model * afr_reduced_cnn_size
        print(f"  Epoch feature dim: {self.epoch_feature_dim}")

        # 特征压缩
        self.feature_compress = nn.Sequential(
            nn.Linear(self.epoch_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # 时序建模（跨epoch）
        self.temporal_transformer = TemporalTransformer(
            d_model=256,
            n_heads=4,
            n_layers=2,
            dropout=0.1
        )

        # 分类头（为每个epoch预测）
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        """
        Args:
            x: (batch, epochs_per_window, signal_length)
        Returns:
            output: (batch, epochs_per_window, num_classes)
        """
        batch_size, num_epochs, signal_length = x.shape

        # 1. 对每个epoch独立提取特征
        # 重塑为 (batch * num_epochs, 1, signal_length)
        x_reshaped = x.view(batch_size * num_epochs, 1, signal_length)

        # MRCNN特征提取
        epoch_features = self.mrcnn(x_reshaped)  # (batch * num_epochs, afr_size, feature_len)

        # TCE注意力
        epoch_features = self.tce(epoch_features)  # (batch * num_epochs, afr_size, d_model)

        # 展平
        epoch_features = epoch_features.contiguous().view(batch_size * num_epochs, -1)
        # (batch * num_epochs, epoch_feature_dim)

        # 特征压缩
        epoch_features = self.feature_compress(epoch_features)  # (batch * num_epochs, 256)

        # 重塑为序列
        epoch_features = epoch_features.view(batch_size, num_epochs, -1)
        # (batch, num_epochs, 256)

        # 2. 时序建模（跨epoch依赖）
        temporal_features = self.temporal_transformer(epoch_features)
        # (batch, num_epochs, 256)

        # 3. 为每个epoch分类
        output = self.classifier(temporal_features)
        # (batch, num_epochs, num_classes)

        return output

    def get_epoch_predictions(self, x):
        """获取每个epoch的预测概率"""
        output = self.forward(x)
        probs = F.softmax(output, dim=-1)
        return probs


def test_model():
    """测试不同窗口长度的模型"""
    print("Testing Short Window EEG AttnSleep Models...")

    for window_min in [3, 5, 10, 30]:
        print(f"\n{'=' * 60}")
        print(f"Testing {window_min}-minute window model")
        print('=' * 60)

        model = ShortWindowAttnSleep(window_minutes=window_min)

        # 测试输入
        # 假设EEG信号长度为3000（30秒 * 100Hz）
        signal_length = 3000
        epochs_per_window = window_min * 2

        x = torch.randn(4, epochs_per_window, signal_length)

        print(f"Input shape: {x.shape}")

        # 前向传播
        try:
            output = model(x)
            print(f"Output shape: {output.shape}")
            print(f"Expected: (4, {epochs_per_window}, 4)")

            # 参数量统计
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

        # 清理
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nAll tests completed!")


if __name__ == "__main__":
    test_model()