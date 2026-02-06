import torch
import torch.nn as nn
from ..conv import Conv
from .routers import CrossModalRouter
from .experts import DualModalExpertContainer

class C2f_DualModal_MoE(nn.Module):
    """
    基于 C2f 改造的双模态 MoE 模块
    参数: c1(输入), c2(输出), n(数量-这里简化为1), shortcut, g, e, num_experts, top_k
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, num_experts=4, top_k=2, Layer_id='C2f_DualModal_MoE'):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels

        # 1x1 卷积做特征变换和降维/升维
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)

        # --- MOE 核心组件 ---
        # 注意：这里的输入通道是 self.c (隐藏层通道数)
        self.router = CrossModalRouter(self.c, num_experts, top_k=top_k, Layer_id=f'{Layer_id}_Router')
        self.experts = DualModalExpertContainer(self.c, self.c)

        # 占位符，保持 C2f 结构完整性，但实际计算由 experts 接管
        self.m = nn.ModuleList(nn.Identity() for _ in range(n))

    def forward(self, x):
        # 1. C2f 的分流操作
        y = list(self.cv1(x).split((self.c, self.c), 1))

        # 2. 取其中一支送入 MoE
        # y[-1] 此时包含了 RGB和IR 的融合特征（由 cv1 混过的）
        # 但我们假设在特征空间中，前一半依然主要由 RGB 贡献，后一半由 IR 贡献
        # 或者在 cv1 使用 group conv 来强制隔离（可选优化）
        moe_input = y[-1]

        # 3. 路由 + 专家计算
        r_weights, r_indices = self.router(moe_input)
        expert_out = self.experts(moe_input, r_weights, r_indices)

        # 4. 替换原来的 Bottleneck 输出
        y.extend(m(expert_out) for m in self.m) # 这里 m 是 Identity

        # 5. 最终融合
        return self.cv2(torch.cat(y, 1))