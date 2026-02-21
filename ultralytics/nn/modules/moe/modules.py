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
    修复版通用MoE容器：
    1. 专家初始化传入expert_id，实现差异化；
    2. 强化index_add_鲁棒性，适配Top-K>1；
    3. 兼容YOLO的2D特征格式（B,C,H,W）。
    """
    def __init__(self, in_channels, out_channels, num_experts=4, top_k=1, expand_ratio=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.out_channels = out_channels
        self.expand_ratio = expand_ratio

        # ✅ 核心修复：初始化专家时传入expert_id，实现差异化
        self.experts = nn.ModuleList([
            OptimizedSimpleExpert(
                in_channels=in_channels,
                out_channels=out_channels,
                expert_id=i,  # 每个专家传入专属ID
                expand_ratio=expand_ratio
            ) for i in range(num_experts)
        ])

    def forward(self, x, weights, indices):
        """
        稀疏推理逻辑（优化鲁棒性）：
        x: [B, C, H, W]
        weights: [B, TopK]
        indices: [B, TopK]
        """
        B, C, H, W = x.shape
        expert_output = torch.zeros(B, self.out_channels, H, W, device=x.device, dtype=x.dtype)

        # 展平索引/权重（兼容Top-K>1）
        indices_flat = indices.view(-1)  # [B*TopK]
        weights_flat = weights.view(-1)  # [B*TopK]

        # 遍历每个专家，稀疏聚合输出
        for expert_id, expert in enumerate(self.experts):
            # 找到选中当前专家的flat索引
            mask = (indices_flat == expert_id)
            if not mask.any():  # 无样本选中当前专家，跳过（更简洁的判断）
                continue

            # 反算batch索引（Top-K>1时正确）
            batch_indices = torch.div(mask.nonzero(as_tuple=True)[0], self.top_k, rounding_mode='floor')
            # 提取选中样本的输入
            selected_x = x[batch_indices]
            # 专家前向（GroupNorm适配小Batch）
            expert_out = expert(selected_x)
            # 加权（权重维度扩展到[Num,1,1,1]，匹配特征）
            selected_weights = weights_flat[mask].view(-1, 1, 1, 1)
            weighted_out = expert_out * selected_weights
            # 原位聚合（PyTorch高效稀疏操作）
            expert_output.index_add_(0, batch_indices, weighted_out)

        return expert_output

class C2f_DualModal_MoE(nn.Module):
    """
    你的顶层调用模块 (需更新以使用上述组件)
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, num_experts=4, top_k=1, loss_weight=2.0, Layer_id='MoE_Router'):
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
        self.router = UltraEfficientRouter(self.c, num_experts, top_k=top_k, loss_weight=loss_weight, Layer_id="{}_{}".format(Layer_id, "Router"))
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