import torch
import torch.nn as nn

class SEBlock(nn.Module):
    """优化的Squeeze-Excitation通道注意力模块（简化版）"""
    def __init__(self, channel, reduction=8, use_maxpool=False, gamma_init=1.0):
        super().__init__()
        assert reduction > 0, f"Reduction ratio must be positive, got {reduction}"
        assert channel % reduction == 0, f"Channel({channel}) must be divisible by reduction({reduction})"
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.use_maxpool = use_maxpool
        if use_maxpool:
            self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.gamma = nn.Parameter(torch.tensor(gamma_init))
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        assert x.dim() == 3, f"Input tensor must be 3D (batch, channel, time), got {x.dim()}D"
        b, c, t = x.size()
        y = self.avg_pool(x)
        if self.use_maxpool:
            y = y + self.max_pool(x)
        y = y.view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * (self.gamma * y) + (1 - self.gamma) * x