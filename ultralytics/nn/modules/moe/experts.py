import torch
import torch.nn as nn

def get_safe_groups(channels: int, desired_groups: int = 8) -> int:
    groups = min(desired_groups, channels)
    while channels % groups != 0:
        groups -= 1
    return max(1, groups)

class OptimizedSimpleExpert(nn.Module):
    """
    Backbone 专用专家：统一结构，拒绝人为制造差异导致的分布撕裂
    """
    def __init__(self, in_channels, out_channels, expand_ratio=2, num_groups=8):
        super().__init__()
        hidden_dim = int(in_channels * expand_ratio)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, 1, 1, bias=False),
            nn.GroupNorm(get_safe_groups(hidden_dim, num_groups), hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.GroupNorm(get_safe_groups(out_channels, num_groups), out_channels)
        )

    def forward(self, x):
        return self.conv(x)