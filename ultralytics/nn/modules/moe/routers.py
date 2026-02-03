import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalRouter(nn.Module):
    """
    双模态互注意力路由器：
    1. 接收拼接后的特征 (C_rgb + C_ir)
    2. 拆分并进行互注意力 (Cross-Attention)
    3. 输出专家权重
    """
    def __init__(self, in_channels, num_experts, top_k=2, reduction=16):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts

        # 假设输入是 RGB和IR 通道拼接，所以对半切
        self.c_split = in_channels // 2

        # 全局感知 GAP
        self.gap = nn.AdaptiveAvgPool2d(1)

        # 互注意力交互层 (轻量化 MLP 模拟 Attention)
        self.fc_rgb = nn.Linear(self.c_split, self.c_split // reduction, bias=False)
        self.fc_ir = nn.Linear(self.c_split, self.c_split // reduction, bias=False)

        # 决策层
        fused_dim = (self.c_split // reduction) * 2
        self.router_head = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.LeakyReLU(0.1, inplace=True),
            # 输出: 专家数 * 输入通道数 (生成通道级注意力权重)
            nn.Linear(fused_dim, num_experts * in_channels)
        )

    def forward(self, x):
        b, c, _, _ = x.shape

        # 1. 拆分模态 (假设是 Concat 进来的)
        x_rgb, x_ir = torch.split(x, self.c_split, dim=1)

        # 2. 压缩特征
        v_rgb = self.gap(x_rgb).view(b, -1)
        v_ir = self.gap(x_ir).view(b, -1)

        # 3. 互注意力交互
        f_rgb = self.fc_rgb(v_rgb)
        f_ir = self.fc_ir(v_ir)
        f_fused = torch.cat([f_rgb, f_ir], dim=1)

        # 4. 计算 Logits
        logits = self.router_head(f_fused).view(b, self.num_experts, c)

        # 5. 选 Top-K 专家
        expert_scores = logits.mean(dim=2) # 按通道平均，得到专家总分
        _, topk_indices = torch.topk(expert_scores, self.top_k, dim=1)

        # 6. 生成权重
        if self.training:
            # 软路由：保留梯度
            all_weights = torch.sigmoid(logits)
            gather_indices = topk_indices.unsqueeze(2).expand(-1, -1, c)
            selected_weights = torch.gather(all_weights, 1, gather_indices)
            return selected_weights, topk_indices
        else:
            # 硬路由：极致省电，权重全置 1 (依靠专家的 mask 跳过计算)
            selected_weights = torch.ones(b, self.top_k, c, device=x.device)
            return selected_weights, topk_indices