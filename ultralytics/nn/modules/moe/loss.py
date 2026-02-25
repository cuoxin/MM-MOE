import torch
import torch.nn as nn
import torch.nn.functional as F

class LoadBalancingLoss(nn.Module):
    def __init__(self, num_experts, top_k=1, loss_weight=0.01, z_loss_weight=1e-3, ignore_bg_expert=False):
        """
        loss_weight 默认为 0.01: 这是一个“弹性约束”。
        只要没有专家彻底饿死，Loss 就会很小；只有发生严重坍塌时才提供拉力，不强求 25% 的绝对平均。
        """
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.loss_weight = loss_weight
        self.z_loss_weight = 0
        self.ignore_bg_expert = ignore_bg_expert
        # 增加一个内部计数器，用于计算当前调用次数
        self.register_buffer("step_count", torch.zeros(1, dtype=torch.long))

        # 假设 batch=64，数据集一轮大概多少步？
        # 你可以根据实际情况把 warmup_steps 设置为你前 3 到 5 个 Epoch 的总 iteration 数
        self.warmup_steps = 1500

    def forward(self, router_logits, expert_indices):

        # 计数器累加
        if self.training:
            self.step_count += 1

        # 【杀手锏 2：预热机制】
        # 如果还在预热期内，彻底放飞自我，不计算任何负载均衡
        if self.training and self.step_count.item() < self.warmup_steps:
            return torch.tensor(0.0, device=router_logits.device)

        batch_size = router_logits.size(0)
        K = self.num_experts if not self.ignore_bg_expert else self.num_experts - 1

        # 1. 意图概率 P (可导)
        router_probs = F.softmax(router_logits, dim=-1)
        if self.ignore_bg_expert:
            router_probs = router_probs[:, :-1]
        P_avg = router_probs.mean(dim=0)

        # 2. 真实频率 f (不可导)
        expert_indices_flat = expert_indices.flatten()
        route_one_hot = F.one_hot(expert_indices_flat, num_classes=self.num_experts).float()
        if self.ignore_bg_expert:
            route_one_hot = route_one_hot[:, :-1]

        total_selected = route_one_hot.sum(dim=0)
        total_selection_opportunities = batch_size * self.top_k
        f_avg = (total_selected / total_selection_opportunities).detach()

        # 3. 正确的 Switch 负载损失，配合较小的 loss_weight
        sum_Pf = (P_avg * f_avg).sum()
        balance_loss = (K * sum_Pf - 1).clamp(min=0.0) * self.loss_weight

        # 4. Z-Loss (稳定 Logits，防止数值爆炸)
        log_sum_exp = torch.logsumexp(router_logits, dim=1)
        z_loss = torch.mean(log_sum_exp ** 2) * self.z_loss_weight

        return balance_loss + z_loss