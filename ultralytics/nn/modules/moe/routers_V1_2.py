import torch
import torch.nn as nn
import torch.nn.functional as F
from .stats import MoEStatsRecorder
from .loss import LoadBalancingLoss
from .collector import MoEAuxCollector

class CrossModalRouter(nn.Module):
    """
    双模态互注意力路由器：
    1. 接收拼接后的特征 (C_rgb + C_ir)
    2. 拆分并进行互注意力 (Cross-Attention)
    3. 输出专家权重
    """
    def __init__(self, in_channels, num_experts, top_k=2, reduction=16, Layer_id='MoE_Router'):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.Layer_id = Layer_id
        self.stats_recorder = MoEStatsRecorder()

        self.register_buffer("aux_loss", torch.zeros(1), persistent=False)
        # self.aux_loss = torch.tensor(0.0, device='cuda')  # 存储当前批次的负载均衡损失
        self.balance_loss_fn = LoadBalancingLoss(num_experts, loss_weight=0.5)

        # 假设输入是 RGB和IR 通道拼接，所以对半切
        self.c_split = in_channels // 2

        # 全局感知 GAP
        self.gap = nn.AdaptiveAvgPool2d(1)

        # 互注意力交互层 (轻量化 MLP 模拟 Attention)
        # self.fc_rgb = nn.Linear(self.c_split, self.c_split // reduction, bias=False)
        # self.fc_ir = nn.Linear(self.c_split, self.c_split // reduction, bias=False)

        # 决策层
        # fused_dim = (self.c_split // reduction) * 2
        # self.router_head = nn.Sequential(
        #     nn.Linear(fused_dim, fused_dim),
        #     nn.LeakyReLU(0.1, inplace=True),
        #     # 输出: 专家数 * 输入通道数 (生成通道级注意力权重)
        #     nn.Linear(fused_dim, num_experts * in_channels)
        # )

        # 监测数据初始化
        self.register_buffer("selection_states", torch.zeros(num_experts), persistent=False)
        self.register_buffer("expert_scores_sum", torch.zeros(num_experts), persistent=False)
        self.register_buffer("states_step_count", torch.zeros(1), persistent=False)

        mid_channels = max(16, in_channels // reduction)
        self.router_head = nn.Sequential(
            nn.Linear(in_channels, mid_channels),
            nn.SiLU(inplace=True),
            nn.Linear(mid_channels, num_experts)
        )

    def forward(self, x):
        x_in = x.clone()

        b, c, h, w = x_in.shape

        # 提取全局特征
        global_feat = self.gap(x_in).flatten(1).clone()

        # 计算 Logits
        # logits = self.router_head(f_fused).view(b, self.num_experts, c)
        # logits = self.router_head(f_fused).reshape(b, self.num_experts, c).clone()
        logits = self.router_head(global_feat)

        # 训练时注入噪声
        if self.training:
            noise = torch.randn_like(logits) * 2.0
            noisy_logits = logits + noise
        else:
            noisy_logits = logits

        # 选 Top-K 专家
        _, topk_indices = torch.topk(noisy_logits, self.top_k, dim=1)

        # 生成权重
        if self.training:

            # 数据监测区
            with torch.no_grad():
                # 统计选择次数
                flat_indices = topk_indices.flatten()
                counts = torch.bincount(flat_indices, minlength=self.num_experts)
                self.selection_states += counts

                # 统计专家分数 (使用原始 logits 的均值作为分数)
                self.expert_scores_sum += logits.mean(dim=0)
                self.states_step_count += 1

            # 软路由：保留梯度
            topk_logits = torch.gather(logits, 1, topk_indices)
            selected_weights = F.softmax(topk_logits, dim=1)

            # 计算负载损失，使用原始 logits
            aux_loss = self.balance_loss_fn(logits, topk_indices)
            MoEAuxCollector.add(aux_loss)

            return selected_weights, topk_indices
        else:
            self.aux_loss = torch.tensor(0.0, device=x.device)  # 推理阶段不计算负载均衡损失
            # 硬路由：极致省电，权重全置 1 (依靠专家的 mask 跳过计算)
            selected_weights = torch.ones(b, self.top_k, device=x.device)

            return selected_weights, topk_indices