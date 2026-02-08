import torch
import torch.nn as nn

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class DSConv(nn.Module):
    """
    深度可分离卷积 (Depthwise Separable Conv)
    参数量极少，速度快，适合作为轻量级专家
    """
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        self.dw = nn.Conv2d(c1, c1, k, s, autopad(k), groups=c1, bias=False)
        self.pw = nn.Conv2d(c1, c2, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))

class DualModalExpertContainer(nn.Module):
    """
    精简版专家容器
    Expert 0: RGB 专属 (DSConv 3x3)
    Expert 1: IR 专属 (DSConv 3x3)
    Expert 2: 融合 (Conv 1x1)
    Expert 3: 背景 (跳过)
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 假设输入 channels 是 concat 后的，所以单模态通道是 一半
        self.c_split = in_channels // 2

        self.experts = nn.ModuleList([
            # === Expert 0: RGB 专家 ===
            # 只看前一半通道，用 DW 卷积提取细节
            DSConv(self.c_split, out_channels, k=3),

            # === Expert 1: IR 专家 ===
            # 只看后一半通道，用 DW 卷积提取轮廓
            DSConv(self.c_split, out_channels, k=3),

            # === Expert 2: 融合专家 ===
            # 看所有通道，用 1x1 卷积快速融合，不改变空间信息
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.SiLU()
            ),

            # === Expert 3: 背景专家 ===
            # 占位符，实际计算时直接跳过
            nn.Identity()
        ])

    def forward(self, x, weights, indices):
        """
        x: [B, C, H, W]
        weights: [B, TopK]  (Routers 返回的是 [B, TopK])
        indices: [B, TopK]
        """
        b, c, h, w = x.shape
        # 初始化输出，默认为 0 (这就实现了“背景专家输出全0”的效果)
        out = torch.zeros(b, self.experts[0].pw.out_channels, h, w, device=x.device)

        # 为了加速，我们先切分好 RGB 和 IR 数据
        # 避免在循环里切分，虽然 PyTorch 的切片是视图，但尽量减少操作
        x_rgb = x[:, :self.c_split]
        x_ir = x[:, self.c_split:]

        # 遍历 Top-K (K 通常很小，比如 2)
        for k in range(indices.shape[1]):
            # 当前这一轮，Batch 中每个样本选择了哪个专家
            idx_k = indices[:, k] # [B]
            w_k = weights[:, k]   # [B]

            # === 优化逻辑：向量化掩码操作 ===

            # --- 处理 Expert 0 (RGB) ---
            # 找到本轮选了 RGB 专家的样本索引
            mask_0 = (idx_k == 0)
            if mask_0.any():
                # 只计算 mask 为 True 的那部分样本
                subset_x = x_rgb[mask_0]
                expert_out = self.experts[0](subset_x)
                # 加权累加: out += expert_out * weight
                # view(-1, 1, 1, 1) 是为了广播权重到 [Batch_subset, 1, 1, 1]
                out[mask_0] += expert_out * w_k[mask_0].view(-1, 1, 1, 1)

            # --- 处理 Expert 1 (IR) ---
            mask_1 = (idx_k == 1)
            if mask_1.any():
                subset_x = x_ir[mask_1]
                expert_out = self.experts[1](subset_x)
                out[mask_1] += expert_out * w_k[mask_1].view(-1, 1, 1, 1)

            # --- 处理 Expert 2 (Fusion) ---
            mask_2 = (idx_k == 2)
            if mask_2.any():
                subset_x = x[mask_2] # 融合专家需要全量输入
                expert_out = self.experts[2](subset_x)
                out[mask_2] += expert_out * w_k[mask_2].view(-1, 1, 1, 1)

            # --- 处理 Expert 3 (Background) ---
            # 直接跳过，什么都不做，out 保持原值（如果是唯一的专家，out就是0）
            # 这就是你要的“空背景直接跳过”

        return out