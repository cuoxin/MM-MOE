import torch
import torch.nn as nn
import torch.nn.functional as F

class LoadBalancingLoss(nn.Module):
    def __init__(self, num_experts, top_k, loss_weight=0.05, z_loss_weight=1e-3, ignore_bg_expert=False):
        """
        Args:
            num_experts: 专家总数
            loss_weight: 负载均衡主权重 (建议 0.1~2.0，根据训练情况调整)
            z_loss_weight: 稳定 Logits 的 Z-Loss 权重 (默认 1e-3 即可)
            ignore_bg_expert: 是否开启“特化背景专家”。若为 True，第 0 号专家将不受均分惩罚。
        """
        super().__init__()
        self.num_experts = num_experts
        self.loss_weight = loss_weight
        self.top_k = top_k
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
        # 修复：正确的 Z-Loss (加入log(num_experts)基线)
        # ==========================================
        log_sum_exp = torch.logsumexp(router_logits, dim=1)
        z_loss = torch.mean((log_sum_exp - torch.log(torch.tensor(self.num_experts, device=log_sum_exp.device))) ** 2)
        z_loss = z_loss * self.z_loss_weight

        # ==========================================
        # 修复：正确的 Switch Transformer 均衡 Loss (加入1 - ，优化方向正确)
        # ==========================================
        # 1. 计算 Router 意图 (Importance, 可导)
        router_probs = F.softmax(router_logits, dim=1)
        P_avg = router_probs.mean(dim=0)  # [Num_Experts]

        # 2. 统计真实物理负载 (Load, 不可导)
        route_one_hot = F.one_hot(expert_indices, num_classes=self.num_experts).float()
        f_avg = route_one_hot.sum(dim=(0, 1)) / (batch_size * self.top_k)
        f_avg = f_avg.detach()  # 核心：真实负载不参与梯度计算

        # ==========================================
        # 修复：背景专家处理（移除冗余归一化，优化约束逻辑）
        # ==========================================
        if self.ignore_bg_expert and self.num_experts > 1:
            # 切片剔除 Exp0，只约束目标专家 (Exp1, Exp2, Exp3)
            P_avg_target = P_avg[1:]
            f_avg_target = f_avg[1:]  # ✅ 移除错误的归一化

            num_target_experts = self.num_experts - 1
            # ✅ 加入1 - ，优化方向正确
            balance_loss = (1 - (num_target_experts * (P_avg_target * f_avg_target).sum())) * self.loss_weight
        else:
            # 标准全员约束（加入1 - ）
            balance_loss = (1 - (self.num_experts * (P_avg * f_avg).sum())) * self.loss_weight

        # 最终总辅助损失（确保损失非负，避免梯度震荡）
        total_aux_loss = balance_loss.clamp(min=0.0) + z_loss

        return total_aux_loss