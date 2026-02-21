# experts.py
import torch
import torch.nn as nn

def get_safe_groups(channels: int, desired_groups: int = 8) -> int:
    groups = min(desired_groups, channels)
    while channels % groups != 0:
        groups -= 1
    return max(1, groups)

class OptimizedSimpleExpert(nn.Module):
    """
    适配UniversalMoEContainer的差异化专家：
    1. 加入expert_id实现初始化/分组/特征偏移差异化；
    2. 完全兼容YOLO的Conv模块通道格式；
    3. GroupNorm适配小Batch，避免MoE稀疏推理的归一化抖动。
    """
    def __init__(self, in_channels, out_channels, expert_id, expand_ratio=2, base_num_groups=8):
        super().__init__()
        self.expert_id = expert_id  # 专家专属ID，核心差异化标识
        self.in_channels = in_channels
        self.out_channels = out_channels
        hidden_dim = int(in_channels * expand_ratio)

        # 1. 差异化分组数（每个专家分组不同，打破归一化趋同）
        self.num_groups = base_num_groups + expert_id  # Exp0:8, Exp1:9, Exp2:10, Exp3:11
        self.hidden_groups = get_safe_groups(hidden_dim, self.num_groups)
        self.out_groups = get_safe_groups(out_channels, self.num_groups)

        # 2. 保留3x3卷积提取空间特征（适配Backbone/C2f）
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, 1, 1, bias=False),
            nn.GroupNorm(self.hidden_groups, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.GroupNorm(self.out_groups, out_channels)
        )

        # 3. 专家专属特征偏移（微小，不影响性能，仅打破初始趋同）
        self.register_buffer("offset", torch.randn(1, out_channels, 1, 1) * 0.01)

        # 4. 差异化初始化（每个专家参数不同）
        self._init_weights()

    def _init_weights(self):
        """专属初始化：不同专家用不同种子+参数偏移"""
        torch.manual_seed(100 + self.expert_id)  # 专属种子
        for m in self.conv.modules():
            if isinstance(m, nn.Conv2d):
                # 权重均值随专家ID偏移，强化初始差异
                mean = 0.02 * (self.expert_id % 3 - 1)
                nn.init.normal_(m.weight, mean=mean, std=0.01)
        torch.manual_seed(torch.initial_seed())  # 恢复全局种子

    def forward(self, x):
        # 前向+专属偏移，确保初始输出差异化
        out = self.conv(x) + self.offset
        return out