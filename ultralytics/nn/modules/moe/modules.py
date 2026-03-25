import torch
import torch.nn as nn
from .routers import UltraEfficientRouter
from .experts import DecoupledMoEContainer, PassThroughExpert
from ..conv import Conv


class C2f_DualModal_MoE(nn.Module):
    def __init__(self, c1, c2, n=1, num_experts=4, top_k=1, shared_experts_nums=1, pass_through_expert_nums=1, decay_steps=1000, Layer_id=None):
        super().__init__()

        # print(f"\n{'='*50}")
        # print(f"🛠️ [调试] 正在初始化 C2f_DualModal_MoE (Layer: {Layer_id})")
        # print(f"  ➔ c1 (输入通道 - 由上一层自动推导): {c1}")
        # print(f"  ➔ c2 (输出通道 - YAML中传入的第一个数字): {c2}")
        # print(f"  ➔ num_experts (路由专家总数): {num_experts}")
        # print(f"  ➔ top_k (激活专家数量): {top_k}")
        # print(f"  ➔ shared_experts_nums (共享专家数): {shared_experts_nums}")
        # print(f"  ➔ pass_through_expert_nums (直通专家数): {pass_through_expert_nums}")
        # print(f"  ➔ decay_steps (噪声退火步数): {decay_steps}")
        # print(f"{'='*50}\n")

        self.input_channels = c1
        self.output_channels = c2

        self.router = UltraEfficientRouter(self.input_channels,
                                            num_routed_experts=num_experts,
                                            top_k=top_k,
                                            pass_through_expert_nums=pass_through_expert_nums,
                                            loss_weight=0.005,
                                            decay_steps=decay_steps,
                                            Layer_id="{}_{}".format(Layer_id, "Router")
                                            )
        self.experts = DecoupledMoEContainer(self.input_channels,
                                            self.output_channels,
                                            num_experts=num_experts,
                                            top_k=top_k,
                                            shared_experts_nums=shared_experts_nums,
                                            pass_through_expert_nums=pass_through_expert_nums)

    def forward(self, x):

        router_weights, router_indices = self.router(x)
        moe_out = self.experts(x, router_weights, router_indices)

        return moe_out