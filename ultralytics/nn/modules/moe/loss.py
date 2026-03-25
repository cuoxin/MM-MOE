import torch
import torch.nn as nn
import torch.nn.functional as F

class LoadBalancingLoss(nn.Module):
    # 🐛 将 ignore_bg_expert 改为 bg_num (背景专家数量)
    def __init__(self, num_experts, top_k=1, loss_weight=100, z_loss_weight=1e-3, bg_num=0):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.loss_weight = loss_weight
        self.z_loss_weight = z_loss_weight
        self.bg_num = bg_num  # 记录背景专家的具体数量

    def forward(self, router_logits, expert_indices):
        batch_size = router_logits.size(0)

        # 1. 意图概率 P (可导)
        router_probs = F.softmax(router_logits, dim=-1)

        # 2. 真实频率 f (不可导)
        expert_indices_flat = expert_indices.flatten()
        route_one_hot = F.one_hot(expert_indices_flat, num_classes=self.num_experts).float()

        # 🛡️ 完美防御：根据实际直通专家数量动态切片
        if self.bg_num > 0:
            # 切除前 bg_num 个直通专家
            router_probs = router_probs[:, self.bg_num:]
            route_one_hot = route_one_hot[:, self.bg_num:]

            router_probs = router_probs / (router_probs.sum(dim=-1, keepdim=True) + 1e-6)
            P_avg = router_probs.mean(dim=0)

            total_selected = route_one_hot.sum(dim=0)
            total_selection_opportunities = total_selected.sum() + 1e-6
            f_avg = (total_selected / total_selection_opportunities).detach()

            K = self.num_experts - self.bg_num
        else:
            P_avg = router_probs.mean(dim=0)
            total_selected = route_one_hot.sum(dim=0)
            total_selection_opportunities = batch_size * self.top_k
            f_avg = (total_selected / total_selection_opportunities).detach()
            K = self.num_experts

        # 3. 正确的 Switch 负载损失
        sum_Pf = (P_avg * f_avg).sum()
        balance_loss = (K * sum_Pf - 1.0).clamp(min=0.0) * self.loss_weight

        # 4. Z-Loss (稳定 Logits)
        log_sum_exp = torch.logsumexp(router_logits, dim=1)
        z_loss = torch.mean(log_sum_exp ** 2) * self.z_loss_weight

        # if self.training:
        #     print(f"[Debug] LoadBalancingLoss -> balance_loss: {balance_loss.item():.6f}, z_loss: {z_loss.item():.6f}")

        return balance_loss + z_loss