import torch
import torch.nn as nn
from .routers import UltraEfficientRouter
from .experts import OptimizedSimpleExpert # 请确保这里对应你原本使用的极简专家类名
from ..conv import Conv

class DecoupledMoEContainer(nn.Module):
    def __init__(self, in_channels, out_channels, num_experts=4, top_k=1):
        super().__init__()
        self.top_k = top_k
        self.out_channels = out_channels

        # 🌟 创新点 2：通用与特化解耦 (4个专家 = 1个共享 + 3个路由)
        self.num_routed_experts = num_experts - 1

        # 共享专家 (不参与路由，兜底全图宏观特征)
        self.shared_expert = OptimizedSimpleExpert(in_channels, out_channels)

        # 路由专家池 (处理 Router 丢过来的特化微观特征)
        self.routed_experts = nn.ModuleList([
            OptimizedSimpleExpert(in_channels, out_channels)
            for _ in range(self.num_routed_experts)
        ])

    def forward(self, x, weights, indices):
        # [分支 1] 共享专家永远激活，提取通用特征
        shared_out = self.shared_expert(x)

        # [分支 2] 路由专家按需激活，提取特化特征
        routed_out = torch.zeros_like(shared_out)
        indices_flat = indices.view(-1)
        weights_flat = weights.view(-1)

        for i, expert in enumerate(self.routed_experts):
            mask = (indices_flat == i)
            if not mask.any():
                continue

            batch_indices = mask.nonzero(as_tuple=True)[0]
            selected_x = x[batch_indices]
            expert_out = expert(selected_x)

            selected_weights = weights_flat[mask].view(-1, 1, 1, 1)
            weighted_out = expert_out * selected_weights

            routed_out.index_add_(0, batch_indices, weighted_out)

        # [完美融合] 通用特征 + 特化特征
        return shared_out + routed_out

class C2f_DualModal_MoE(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, num_experts=4, top_k=1, Layer_id='MoE_Router'):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)

        # 注意：这里传给 Router 的专家数量是 num_experts - 1
        num_routed = num_experts - 1
        self.router = UltraEfficientRouter(self.c, num_routed_experts=num_routed, top_k=top_k, Layer_id="{}_{}".format(Layer_id, "Router"))
        self.experts = DecoupledMoEContainer(self.c, self.c, num_experts=num_experts, top_k=top_k)

        self.m = nn.ModuleList(nn.Identity() for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))

        # 路由前向传播，彻底抛弃了 aux_loss 的纠缠
        router_weights, router_indices = self.router(y[-1])
        moe_out = self.experts(y[-1], router_weights, router_indices)

        y.extend(m(moe_out) for m in self.m)
        return self.cv2(torch.cat(y, 1))