import torch
import torch.nn as nn
import torch.nn.functional as F

class LoadBalancingLoss(nn.Module):
    def __init__(self, num_experts, loss_weight=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.loss_weight = loss_weight

    def forward(self, router_logits, expert_indices):
        """
        Args:
            router_logits: [Batch, Num_Experts] (或者是 [Batch, Num_Experts, Channels] 被平均后的)
                           这是 Router 输出的原始分数 (Softmax之前)
            expert_indices: [Batch, TopK] 选中的专家索引 (仅用于调试或Hard Loss，Aux Loss主要用概率)
        """
        # 1. 重要：计算概率分布 (P)
        # Softmax 必须在 dim=1 上进行
        router_probs = F.softmax(router_logits, dim=1)

        # 2. 专家的平均概率 (P_avg) - 这个代表 Router "想" 选谁
        # [Num_Experts]
        P_avg = router_probs.mean(dim=0)

        # 3. 专家的实际负载 (f) - 这个代表 实际 "选" 了谁
        # 为了可导，我们通常用 Softmax 的输出来近似实际选择，
        # 或者直接计算 load_balancing_loss = num_experts * sum(P_i * f_i)
        # 在 Switch Transformer 中，f_i 也是通过 probs 计算的，
        # 简单版：f_i = fraction of samples dispatched to expert i.

        # 这里的实现技巧：
        # 我们不使用不可导的 argmax/topk indices 来算 f，
        # 而是直接使用 router_probs 来计算“软负载”。
        # 这样整个 Loss 都是平滑可导的，且完全依赖于 router_logits。

        # 实际负载 f_i (Soft Approximation)
        f_avg = router_probs.mean(dim=0)

        # 4. 计算损失: alpha * N * sum(P_i * f_i)
        # 使得 P 和 f 都趋向于均匀分布 (1/N)
        # 注意：这里我们用 P_avg * P_avg 是一种简化的 Auxiliary Loss，
        # 它鼓励 router_probs 在 batch 上平均分布。
        loss = (self.num_experts * (P_avg * f_avg).sum()) * self.loss_weight

        return loss