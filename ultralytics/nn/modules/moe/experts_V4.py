'''这个版本的所有代码在topk=1时都经过了测试验证，都可以正常跑通。'''

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
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(get_safe_groups(hidden_dim, num_groups), hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(get_safe_groups(out_channels, num_groups), out_channels)
        )

    def forward(self, x):
        return self.conv(x)

class PassThroughExpert(nn.Module):
    """
    🌟 直通专家 (Zero-FLOPs / Identity Expert)
    核心作用：为平坦背景和低频特征提供无损透传通道，避免重火力专家的过度计算放大噪声，从而提升 mAP50-95。
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 如果通道一致，使用真正的 0 参数透传
        if in_channels == out_channels:
            self.proj = nn.Identity()
        else:
            # 如果通道需要对齐，使用极轻量 1x1 卷积
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        return self.proj(x)

class DecoupledMoEContainer(nn.Module):
    def __init__(self, in_channels, out_channels, num_experts=4, top_k=1, shared_experts_nums=1, pass_through_expert_nums=1):
        super().__init__()
        self.top_k = top_k
        self.out_channels = out_channels
        self.num_routed_experts = num_experts
        self.shared_experts_nums = shared_experts_nums
        self.pass_through_expert_nums = pass_through_expert_nums

        # ==========================================
        # [分支 1] 共享专家 (Shared Expert)
        # 负责提取通用基础特征，作为所有路由的“兜底”
        # ==========================================
        if self.shared_experts_nums > 0:
            self.shared_expert = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True)
            )
        else:
            self.shared_expert = None

        # ==========================================
        # [分支 2] 异构路由专家池 (Routed Experts)
        # ==========================================
        self.routed_experts = nn.ModuleList()

        # Expert 0 ~ K-1: 塞入直通专家（走捷径，抗噪降算力）
        for _ in range(self.pass_through_expert_nums):
            self.routed_experts.append(PassThroughExpert(in_channels, out_channels))

        # Expert K ~ N-1: 塞入重火力特化专家（处理多模态复杂特征）
        for _ in range(self.pass_through_expert_nums, self.num_routed_experts):
            self.routed_experts.append(OptimizedSimpleExpert(in_channels, out_channels))

    # 所谓真正的稀疏计算方式，速度会比较慢
    # def forward(self, x, weights, indices):
    #     B, C, H, W = x.shape

    #     if self.shared_expert is not None:
    #         shared_out = self.shared_expert(x)
    #         routed_out = torch.zeros_like(shared_out)
    #     else:
    #         shared_out = torch.zeros((B, self.out_channels, H, W), device=x.device, dtype=x.dtype)
    #         routed_out = torch.zeros((B, self.out_channels, H, W), device=x.device, dtype=x.dtype)

    #     for i, expert in enumerate(self.routed_experts):
    #         batch_mask = (indices == i).any(dim=1)

    #         if not batch_mask.any():
    #             continue

    #         x_selected = x[batch_mask]
    #         w_selected = weights[batch_mask][indices[batch_mask] == i].view(-1, 1, 1, 1)

    #         expert_out = expert(x_selected)
    #         weighted_out = expert_out * w_selected

    #         # 🛡️ 内存修复核心：放弃 routed_out[batch_mask] += weighted_out
    #         # 改用临时 Tensor 填入数据，然后通过常规的加法融合，完全避开底层的 Scatter 内存碎片！
    #         temp_out = torch.zeros_like(routed_out)
    #         temp_out[batch_mask] = weighted_out
    #         routed_out = routed_out + temp_out  # 常规加法，对计算图极其友好

    #     return shared_out + routed_out

    # 旧版本计算方式-2026-03-19
    def forward(self, x, weights, indices):

        self.spatial_indices = indices

        B, _, H, W = x.shape


        # 1. 共享分支计算与零张量安全初始化
        if self.shared_expert is not None:
            shared_out = self.shared_expert(x)
            # 根据 shared_out 的正确形状初始化 routed_out
            routed_out = torch.zeros_like(shared_out)
        else:
            # 如果没有共享专家，必须手动创建 out_channels 维度的零张量
            # 绝对不能用 torch.zeros_like(x)
            shared_out = torch.zeros((B, self.out_channels, H, W), device=x.device, dtype=x.dtype)
            routed_out = torch.zeros((B, self.out_channels, H, W), device=x.device, dtype=x.dtype)

        # 2. 预处理广播维度 [B, 1, 1, 1]
        indices_b = indices.view(-1, 1, 1, 1)
        weights_b = weights.view(-1, 1, 1, 1)


        # 3. 极速密集掩码计算
        for i, expert in enumerate(self.routed_experts):
            # 生成当前专家的乘数矩阵 (选中则为概率，未选中则全为 0)
            multiplier = (indices_b == i).float() * weights_b

            # 硬件执行：所有专家并行运算
            expert_out = expert(x)

            # 数学执行：原地融合乘加 (Fused Multiply-Add)
            # 完全杜绝了维度越界，且不会产生额外的显存碎片
            routed_out.addcmul_(expert_out, multiplier)

        # 4. 完美特征融合 (通用特征 + 特化特征)
        return shared_out + routed_out