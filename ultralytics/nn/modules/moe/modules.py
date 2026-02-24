import torch
import torch.nn as nn
import torch.nn.functional as F

from .routers import UltraEfficientRouter
from .experts import OptimizedSimpleExpert
from ..conv import Conv

class UniversalMoEContainer(nn.Module):
    def __init__(self, in_channels, out_channels, num_experts=4, top_k=1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.out_channels = out_channels

        # 统一初始化，移除 expert_id
        self.experts = nn.ModuleList([
            OptimizedSimpleExpert(in_channels, out_channels)
            for _ in range(num_experts)
        ])

    def forward(self, x, weights, indices):
        B, C, H, W = x.shape
        expert_output = torch.zeros(B, self.out_channels, H, W, device=x.device, dtype=x.dtype)

        indices_flat = indices.view(-1)
        weights_flat = weights.view(-1)

        for i, expert in enumerate(self.experts):
            mask = (indices_flat == i)
            if not mask.any():
                continue

            batch_indices = torch.div(mask.nonzero(as_tuple=True)[0], self.top_k, rounding_mode='floor')
            selected_x = x[batch_indices]
            expert_out = expert(selected_x)

            # 使用 V1_9 修复后的维度扩展方式，更稳定
            selected_weights = weights_flat[mask].view(-1, 1, 1, 1)
            weighted_out = expert_out * selected_weights

            expert_output.index_add_(0, batch_indices, weighted_out)

        return expert_output

class C2f_DualModal_MoE(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, num_experts=4, top_k=1, loss_weight=0.01, Layer_id='MoE_Router'):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)

        # 默认下调 Loss Weight 为 0.01 (弹性约束)
        self.router = UltraEfficientRouter(self.c, num_experts, top_k=top_k, loss_weight=loss_weight, Layer_id="{}_{}".format(Layer_id, "Router"))
        self.experts = UniversalMoEContainer(self.c, self.c, num_experts, top_k)

        self.m = nn.ModuleList(nn.Identity() for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))

        router_weights, router_indices, router_logits = self.router(y[-1])
        moe_out = self.experts(y[-1], router_weights, router_indices)

        y.extend(m(moe_out) for m in self.m)
        return self.cv2(torch.cat(y, 1))