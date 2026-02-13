import torch
import torch.nn as nn

def get_safe_groups(channels: int, desired_groups: int = 8) -> int:
    """Ensure num_groups divides channels"""
    groups = min(desired_groups, channels)
    while channels % groups != 0:
        groups -= 1
    return max(1, groups)

class OptimizedSimpleExpert(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=2, num_groups=8):
        super().__init__()
        hidden_dim = int(in_channels * expand_ratio)
        self.conv = nn.Sequential(
            # === 修改建议：这里改为 3x3 卷积，padding=1 ===
            # 这样专家既有通道混合能力，又有空间感知能力，就像原本的 Bottleneck 一样
            nn.Conv2d(in_channels, hidden_dim, 3, 1, 1, bias=False),

            nn.GroupNorm(get_safe_groups(hidden_dim, num_groups), hidden_dim),
            nn.SiLU(inplace=True),

            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.GroupNorm(get_safe_groups(out_channels, num_groups), out_channels)
        )

    def forward(self, x):
        return self.conv(x)