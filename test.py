import torch
import torch.nn as nn
import torch.nn.functional as F

class LoadBalancingLoss(nn.Module):
    """
    带全量日志的负载均衡损失计算类：
    1. 适配Top-K=2、4专家场景；
    2. 打印每一步计算值，定位损失为0的根源；
    3. 包含z_loss、最大负载惩罚（GShard），对齐业界标准。
    """
    def __init__(self, num_experts=4, top_k=2, loss_weight=0.25,
                 z_loss_weight=1e-3, max_load_penalty_weight=0.5,
                 ignore_bg_expert=False, log_level="debug"):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k  # 显式指定Top-K，避免传参错误
        self.loss_weight = loss_weight
        self.z_loss_weight = z_loss_weight
        self.max_load_penalty_weight = max_load_penalty_weight
        self.ignore_bg_expert = ignore_bg_expert
        self.log_level = log_level  # debug:打印所有值，info:只打印关键值

    def forward(self, router_logits, expert_indices, epoch=0):
        """
        router_logits: [B, num_experts] 路由原始logits
        expert_indices: [B, top_k] 选中的专家索引
        epoch: 当前epoch，用于日志标注
        """
        # ===================== 第一步：提取基础维度 =====================
        batch_size = router_logits.size(0)
        num_experts = self.num_experts if not self.ignore_bg_expert else self.num_experts - 1
        K = num_experts  # 公式中的K（专家数）

        # ===================== 第二步：计算P_avg（路由概率均值，可导） =====================
        # 1. 计算路由概率（softmax）
        router_probs = F.softmax(router_logits, dim=-1)
        # 2. 只保留有效专家（无背景专家则全保留）
        if self.ignore_bg_expert:
            router_probs = router_probs[:, :-1]
        # 3. 计算P_avg（每个专家的平均概率）
        P_avg = router_probs.mean(dim=0)  # [num_experts]

        # ===================== 第三步：计算f_avg（实际选中频率，不可导） =====================
        # 1. 构建one-hot矩阵 [B*top_k, num_experts]
        expert_indices_flat = expert_indices.view(-1)  # [B*top_k]
        route_one_hot = F.one_hot(expert_indices_flat, num_classes=self.num_experts).float()
        # 2. 只保留有效专家
        if self.ignore_bg_expert:
            route_one_hot = route_one_hot[:, :-1]
        # 3. 计算每个专家被选中的总次数
        total_selected = route_one_hot.sum(dim=0)  # [num_experts]
        # 4. 计算f_avg（归一化：总选中次数 / 总选中机会数）
        total_selection_opportunities = batch_size * self.top_k
        f_avg = total_selected / total_selection_opportunities  # [num_experts]

        # ===================== 第四步：计算核心均衡损失 =====================
        # 1. 原始均衡损失（未加权、未截断）
        sum_Pf = (P_avg * f_avg).sum()  # sum(P_avg * f_avg)
        balance_loss_raw = 1 - K * sum_Pf  # 核心公式：1 - K*sum(P_avg*f_avg)
        # 2. 加权 + 截断（避免负数）
        balance_loss = balance_loss_raw.clamp(min=0.0) * self.loss_weight

        # ===================== 第五步：计算z_loss（稳定logits） =====================
        # z_loss = ||logits||^2 * z_loss_weight（业界标准）
        z_loss = torch.sum(router_logits **2) / batch_size * self.z_loss_weight

        # ===================== 第六步：计算最大负载惩罚（防止单专家过载） =====================
        # 1. 计算最大选中频率
        max_f_avg = torch.max(f_avg)
        # 2. 惩罚阈值：4专家Top-2的合理上限是 1/num_experts * 1.2 = 0.25*1.2=0.3
        penalty_threshold = (1.0 / num_experts) * 1.2
        # 3. 计算惩罚（超过阈值的部分）
        max_load_penalty = torch.clamp(max_f_avg - penalty_threshold, min=0.0) * self.max_load_penalty_weight

        # ===================== 第七步：总损失 =====================
        total_aux_loss = balance_loss + z_loss + max_load_penalty

        # ===================== 关键：打印所有计算值（定位问题核心） =====================
        if self.log_level == "debug":
            print(f"\n===== MoE Loss Breakdown (Epoch {epoch}, Top-K={self.top_k}, Num_Experts={num_experts}) =====")
            print(f"1. P_avg（路由平均概率）: {[round(p.item(), 4) for p in P_avg]}")
            print(f"2. f_avg（实际选中频率）: {[round(f.item(), 4) for f in f_avg]}")
            print(f"3. sum(P_avg*f_avg)    : {round(sum_Pf.item(), 4)}")
            print(f"4. K * sum(Pf)         : {round((K * sum_Pf).item(), 4)}")
            print(f"5. balance_loss_raw    : {round(balance_loss_raw.item(), 4)}")
            print(f"6. balance_loss (加权后): {round(balance_loss.item(), 6)}")
            print(f"7. z_loss              : {round(z_loss.item(), 6)}")
            print(f"8. max_f_avg           : {round(max_f_avg.item(), 4)}")
            print(f"9. max_load_penalty    : {round(max_load_penalty.item(), 6)}")
            print(f"10. total_aux_loss     : {round(total_aux_loss.item(), 6)}")
            print("="*80)

        return total_aux_loss

# ===================== 测试代码：模拟你Epoch16的极端场景 =====================
if __name__ == "__main__":
    # 模拟你的场景：4专家、Top-K=2、Epoch16、Layer17_Router的选中率（Exp1=50%，Exp2=2.5%）
    num_experts = 4
    top_k = 2
    batch_size = 100  # 模拟批次大小

    # 1. 模拟router_logits（适配Exp1选中率50%的极端情况）
    # Exp1的logits远高于其他专家，导致P_avg[1]≈0.5
    router_logits = torch.ones(batch_size, num_experts) * (-1.0)  # 初始全为-1
    router_logits[:, 1] = 5.0  # Exp1的logits拉高，模拟路由偏好

    # 2. 模拟expert_indices（Exp1占50%，Exp2=2.5%，Exp0=29.5%，Exp3=18%）
    # 总选中次数：batch_size*top_k = 100*2=200次
    # Exp0: 29.5% → 59次；Exp1:50%→100次；Exp2:2.5%→5次；Exp3:18%→36次
    expert_indices = []
    for _ in range(batch_size):
        # 随机选，但保证整体比例接近你的Epoch16
        if torch.rand(1) < 0.5:
            idx1 = 1  # Exp1
        else:
            idx1 = torch.randint(0, 4, (1,)).item()
        idx2 = torch.randint(0, 4, (1,)).item()
        expert_indices.append([idx1, idx2])
    expert_indices = torch.tensor(expert_indices)  # [100, 2]

    # 3. 初始化损失类并计算
    loss_fn = LoadBalancingLoss(
        num_experts=4,
        top_k=2,
        loss_weight=0.25,
        z_loss_weight=1e-3,
        max_load_penalty_weight=0.5,
        ignore_bg_expert=False,
        log_level="debug"
    )
    total_loss = loss_fn(router_logits, expert_indices, epoch=16)