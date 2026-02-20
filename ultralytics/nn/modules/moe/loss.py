import torch
import torch.nn as nn
import torch.nn.functional as F

class LoadBalancingLoss(nn.Module):
    def __init__(self, num_experts, loss_weight=0.05, z_loss_weight=1e-4, ignore_bg_expert=False):
        """
        Args:
            num_experts: 专家总数
            loss_weight: 负载均衡主权重 (建议 0.01~0.05，不要太大)
            z_loss_weight: 稳定 Logits 的 Z-Loss 权重 (默认 1e-3 即可)
            ignore_bg_expert: 是否开启“特化背景专家”。若为 True，第 0 号专家将不受 25% 均分的惩罚。
        """
        super().__init__()
        self.num_experts = num_experts
        self.loss_weight = 0.001
        self.z_loss_weight = z_loss_weight
        self.ignore_bg_expert = ignore_bg_expert

    def forward(self, router_logits, expert_indices):
        """
        Args:
            router_logits: [Batch, Num_Experts] 未经过 Softmax 的原始分数
            expert_indices: [Batch, TopK] 实际选中的专家索引
        """
        batch_size = router_logits.size(0)
        top_k = expert_indices.size(1)

        # ==========================================
        # 优化点 2：Router Z-Loss (稳定 Logits 不爆炸)
        # ==========================================
        # 计算公式: mean( (log(sum(exp(logits))))^2 )
        # torch.logsumexp 是极其数值安全的写法
        z_loss = torch.mean(torch.logsumexp(router_logits, dim=1) ** 2) * self.z_loss_weight

        # ==========================================
        # 优化点 1：标准的 Switch Transformer 均衡 Loss
        # ==========================================
        # 1. 计算 Router 意图 (Importance, 可导)
        router_probs = F.softmax(router_logits, dim=1)
        P_avg = router_probs.mean(dim=0)  # [Num_Experts]

        # 2. 统计真实物理负载 (Load, 不可导)
        # 将 [Batch, TopK] 展开成 one_hot 矩阵统计次数
        route_one_hot = F.one_hot(expert_indices, num_classes=self.num_experts).float()
        f_avg = route_one_hot.sum(dim=(0, 1)) / (batch_size * top_k)
        f_avg = f_avg.detach() # 核心：真实负载是既定事实，不参与算梯度！

        # ==========================================
        # 优化点 3：支持无人机特化的“背景专家”
        # ==========================================
        if self.ignore_bg_expert and self.num_experts > 1:
            # 切片剔除 Exp0，只约束剩下的目标专家 (Exp1, Exp2, Exp3)
            P_avg_target = P_avg[1:]
            f_avg_target = f_avg[1:]

            # 重新归一化 (防除零)
            target_sum = f_avg_target.sum() + 1e-9
            f_avg_target = f_avg_target / target_sum

            num_target_experts = self.num_experts - 1
            balance_loss = (num_target_experts * (P_avg_target * f_avg_target).sum()) * self.loss_weight
        else:
            # 标准全员约束
            balance_loss = (self.num_experts * (P_avg * f_avg).sum()) * self.loss_weight

        # 最终输出结合了两者的总辅损
        total_aux_loss = balance_loss + z_loss

        return total_aux_loss