import torch
import torch.nn as nn
import torch.nn.functional as F

# 假设你的代码结构中，routers.py 和 experts.py 在同一目录下
# 如果报错，请根据实际路径调整引用，例如 from ultralytics.nn.modules.moe.routers import ...
from .routers import UltraEfficientRouter
from .experts import OptimizedSimpleExpert
from ..conv import Conv # 引用 YOLO 的基础卷积模块，通常用于 C2f 内部

class UniversalMoEContainer(nn.Module):
    """
    通用 MoE 容器：负责管理专家列表和执行稀疏推理 (Sparse Inference)
    """
    def __init__(self, in_channels, out_channels, num_experts=4, top_k=1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.out_channels = out_channels

        # 初始化优化后的 GroupNorm 专家
        self.experts = nn.ModuleList([
            OptimizedSimpleExpert(in_channels, out_channels)
            for _ in range(num_experts)
        ])

    def forward(self, x, weights, indices):
        """
        核心加速逻辑：使用 index_add_ 避免 Python 循环中的低效 Mask 操作
        x: [B, C, H, W]
        weights: [B, TopK]
        indices: [B, TopK]
        """
        B, C, H, W = x.shape
        # 初始化输出张量
        expert_output = torch.zeros(B, self.out_channels, H, W, device=x.device, dtype=x.dtype)

        # 展平索引以便处理
        indices_flat = indices.view(-1) # [B*TopK]
        weights_flat = weights.view(-1) # [B*TopK]

        # 遍历所有专家
        for i, expert in enumerate(self.experts):
            # 1. 找到所有选中当前专家 i 的样本位置 (在 flat 维度上)
            mask_indices = (indices_flat == i).nonzero(as_tuple=True)[0]

            if mask_indices.numel() == 0:
                continue

            # 2. 反算出是哪个 Batch 的数据 (batch_index = flat_index // top_k)
            # 如果 TopK=1，mask_indices 就是 batch_indices
            batch_indices = torch.div(mask_indices, self.top_k, rounding_mode='floor')

            # 3. 提取对应的输入数据 [Num_Selected, C, H, W]
            selected_x = x[batch_indices]

            # 4. 专家前向计算 (GroupNorm 保证了这里即使只有 1 个样本也能稳定计算)
            expert_out = expert(selected_x)

            # 5. 加权
            selected_weights = weights_flat[mask_indices].view(-1, 1, 1, 1)
            weighted_out = expert_out * selected_weights

            # 6. 使用 index_add_ 原位聚合，这是 PyTorch 中最快的稀疏聚合方式之一
            expert_output.index_add_(0, batch_indices, weighted_out)

        return expert_output

class C2f_DualModal_MoE(nn.Module):
    """
    你的顶层调用模块 (需更新以使用上述组件)
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, num_experts=4, top_k=1, Layer_id='MoE_Router'):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1) # n 这里如果是 placeholder 可以设为 0

        # print(f"\n🔍 [MoE Config Check] Layer: {Layer_id}")
        # print(f"   |-- c1 (输入通道): {c1}")
        # print(f"   |-- c2 (输出通道): {c2}")
        # print(f"   |-- n  (堆叠次数): {n}")
        # print(f"   |-- shortcut    : {shortcut}")
        # print(f"   |-- g  (分组数)  : {g}")
        # print(f"   |-- e  (膨胀系数): {e}")
        # print(f"   |-- num_experts : {num_experts}")
        # print(f"   |-- top_k       : {top_k}")

        # === 核心修改：使用新的 Router 和 Container ===
        self.router = UltraEfficientRouter(self.c, num_experts, top_k=top_k, Layer_id="{}_{}".format(Layer_id, "Router"))
        self.experts = UniversalMoEContainer(self.c, self.c, num_experts, top_k)

        # 如果需要保留 C2f 的残差结构，可以在这里添加 Identity
        self.m = nn.ModuleList(nn.Identity() for _ in range(n))

    def forward(self, x):
        # YOLO C2f 的分流逻辑
        y = list(self.cv1(x).chunk(2, 1))

        # 1. 路由计算
        # 注意：这里输入给 router 的是 hidden features (y[-1]) 还是原始 x
        # yolo-master 通常把一部分特征送入 router
        router_weights, router_indices, router_logits = self.router(y[-1])

        # 2. 专家计算 (替换了原有的 bottleneck 计算)
        # 输入是 y[-1] (hidden state)，输出也是 hidden state
        moe_out = self.experts(y[-1], router_weights, router_indices)

        # 将 MoE 输出放回列表 (替换掉原来的部分)
        y.extend(m(moe_out) for m in self.m) # 这里 self.m 是 Identity，直接透传

        # 3. 最终融合
        return self.cv2(torch.cat(y, 1))