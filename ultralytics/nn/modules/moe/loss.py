import torch
import torch.nn as nn
import torch.nn.functional as F

class LoadBalancingLoss(nn.Module):
    def __init__(self, num_experts, top_k=2, loss_weight=0.25,
                 z_loss_weight=1e-3, ignore_bg_expert=False):
        super().__init__()
        self.num_experts = num_experts  # 正确的属性名
        self.top_k = top_k
        self.loss_weight = loss_weight
        self.z_loss_weight = z_loss_weight
        self.ignore_bg_expert = ignore_bg_expert

    def forward(self, router_logits, expert_indices):
        batch_size = router_logits.size(0)
        # 适配ignore_bg_expert：有效专家数 = 总专家数 - (1 if 忽略背景专家 else 0)
        K = self.num_experts if not self.ignore_bg_expert else self.num_experts - 1

        # 1. 计算P_avg（路由平均概率，可导）
        router_probs = F.softmax(router_logits, dim=-1)
        # 忽略背景专家（如果开启）
        if self.ignore_bg_expert:
            router_probs = router_probs[:, :-1]
        P_avg = router_probs.mean(dim=0)

        # 2. 计算f_avg（真实选中频率，不可导，必须detach）
        expert_indices_flat = expert_indices.flatten()  # 修正flatten写法（和你的PyTorch版本兼容）
        # 这里用正确的属性名 self.num_experts
        route_one_hot = F.one_hot(expert_indices_flat, num_classes=self.num_experts).float()
        # 忽略背景专家（如果开启）
        if self.ignore_bg_expert:
            route_one_hot = route_one_hot[:, :-1]
        total_selected = route_one_hot.sum(dim=0)
        total_selection_opportunities = batch_size * self.top_k
        f_avg = (total_selected / total_selection_opportunities).detach()  # 关键：detach防止梯度回传

        # 3. 核心修复：负载均衡损失公式（Top-K场景正确写法）
        sum_Pf = (P_avg * f_avg).sum()
        balance_loss_raw = (K * sum_Pf - 1).clamp(min=0.0)  # 从1-K*sumPf改为K*sumPf-1
        balance_loss = balance_loss_raw * self.loss_weight

        # 4. Z loss（稳定路由logits）
        log_sum_exp = torch.logsumexp(router_logits, dim=1)
        z_loss = torch.mean(log_sum_exp ** 2) * self.z_loss_weight

        # 5. 总辅助损失
        total_aux_loss = balance_loss + z_loss

        # 可选：保留日志打印（方便你看修复后的loss值）
        # if self.training and batch_size > 0:
        #     print(f"\n===== MoE Loss Fix Verify =====")
        #     print(f"sum(P_avg*f_avg)    : {sum_Pf.item():.4f}")
        #     print(f"K * sum(Pf)         : {(K * sum_Pf).item():.4f}")
        #     print(f"balance_loss_raw    : {balance_loss_raw.item():.4f}")
        #     print(f"balance_loss (加权后): {balance_loss.item():.6f}")
        #     print(f"z_loss              : {z_loss.item():.6f}")
        #     print(f"total_aux_loss     : {total_aux_loss.item():.6f}")
        #     print("="*40)

        return total_aux_loss