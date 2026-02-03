import torch
import torch.nn as nn
from ..block import Conv, GhostConv # 复用 YOLO 的基础模块

class DualModalExpertContainer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.c_split = in_channels // 2

        # 定义专家库
        self.expert_map = nn.ModuleList([
            # Expert 0: RGB 细节专家 (小核，专注前半段通道)
            nn.Sequential(Conv(self.c_split, out_channels, k=3, s=1)),

            # Expert 1: IR 轮廓专家 (大核 Depthwise，专注后半段通道)
            nn.Sequential(
                nn.Conv2d(self.c_split, self.c_split, 5, 1, 2, groups=self.c_split, bias=False),
                nn.Conv2d(self.c_split, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.SiLU()
            ),

            # Expert 2: 融合专家 (看全部通道)
            GhostConv(in_channels, out_channels, k=3, s=1),

            # Expert 3: 背景/省电专家 (不做任何操作)
            nn.Identity()
        ])

    def forward(self, x, weights, indices):
        b, c, h, w = x.shape
        out = torch.zeros_like(x)

        # 预先拆分，减少索引开销
        x_rgb = x[:, :self.c_split, :, :]
        x_ir = x[:, self.c_split:, :, :]

        # 遍历 Top-K
        for k in range(indices.shape[1]):
            idxs = indices[:, k] # [Batch]
            w_val = weights[:, k] # [Batch, C]

            # === Expert 0: RGB ===
            mask = (idxs == 0)
            if mask.any():
                out[mask] += self.expert_map[0](x_rgb[mask]) * w_val[mask].view(-1, c, 1, 1)

            # === Expert 1: IR ===
            mask = (idxs == 1)
            if mask.any():
                out[mask] += self.expert_map[1](x_ir[mask]) * w_val[mask].view(-1, c, 1, 1)

            # === Expert 2: Fusion ===
            mask = (idxs == 2)
            if mask.any():
                out[mask] += self.expert_map[2](x[mask]) * w_val[mask].view(-1, c, 1, 1)

            # === Expert 3: Identity (Skip) ===
            # 直接跳过计算，这里不加任何东西，或者做恒等映射
            # 如果是残差结构，这里 return 0 即可，外层会有 x + out
            # 这里为了保持特征流不断，如果是纯背景，我们假设它保留原特征（或被抑制）
            mask = (idxs == 3)
            if mask.any():
                out[mask] += x[mask] * 0.1 # 抑制背景

        return out