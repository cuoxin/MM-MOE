import torch
import torch.nn as nn

class UniversalExpert(nn.Module):
    """
    通用专家模块：不再区分模态，处理全量特征。
    使用标准的 Conv-BN-SiLU 结构，便于推理加速和算子融合。
    """
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        # 普通卷积，或者你可以换成 Bottleneck / C3k2 的简化版
        self.conv = nn.Sequential(
            nn.Conv2d(c1, c2, k, s, k // 2, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )

    def forward(self, x):
        return self.conv(x)

class UniversalMoEContainer(nn.Module):
    """
    通用 MoE 容器
    """
    def __init__(self, in_channels, out_channels, num_experts=4):
        super().__init__()
        self.num_experts = num_experts

        # 创建一组结构相同但参数初始化不同的专家
        self.experts = nn.ModuleList([
            UniversalExpert(in_channels, out_channels)
            for _ in range(num_experts)
        ])

    def forward(self, x, weights, indices):
        """
        x: [B, C, H, W]
        weights: [B, TopK]
        indices: [B, TopK]
        """
        b, c, h, w = x.shape
        out = torch.zeros(b, c, h, w, device=x.device)

        # -----------------------------------------------------------
        # 极速推理模式 (针对 TopK=1 的优化)
        # 如果你配置 top_k=1，这将显著提升速度，避免循环
        # -----------------------------------------------------------
        if indices.shape[1] == 1:
            # 展平索引
            flat_idx = indices.view(-1)      # [B]
            flat_w = weights.view(-1, 1, 1, 1) # [B, 1, 1, 1]

            # 这种方法虽然还是有 mask，但避免了多次 slice 输入张量
            # 只有当对应的 expert 被选中时才计算
            for i, expert in enumerate(self.experts):
                mask = (flat_idx == i)
                if mask.any():
                    # 只计算当前专家负责的那部分样本
                    # [N, C, H, W] -> Expert -> [N, C, H, W]
                    subset_out = expert(x[mask])
                    out[mask] = subset_out * flat_w[mask]

            return out

        # -----------------------------------------------------------
        # 通用模式 (TopK > 1，用于训练或追求更高精度)
        # -----------------------------------------------------------
        else:
            for k in range(indices.shape[1]):
                idx_k = indices[:, k]
                w_k = weights[:, k]

                for i, expert in enumerate(self.experts):
                    mask = (idx_k == i)
                    if mask.any():
                        subset_out = expert(x[mask])
                        out[mask] += subset_out * w_k[mask].view(-1, 1, 1, 1)
            return out