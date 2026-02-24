import torch
import torch.nn as nn
import torch.nn.functional as F

from .stats import MoEStatsRecorder
from .loss import LoadBalancingLoss
from .collector import MoEAuxCollector

class UltraEfficientRouter(nn.Module):
    def __init__(self, in_channels, num_experts, top_k=1, reduction=16, loss_weight=0.01, Layer_id='MoE_Router'):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.Layer_id = Layer_id

        # 保留 V1_8 的 DW+PW 卷积结构，保留空间感知能力，拒绝全局池化盲猜
        reduced_channels = max(in_channels // reduction, 4)
        self.router = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(reduced_channels, num_experts)
        )

        self.balance_loss_fn = LoadBalancingLoss(num_experts, top_k=top_k, loss_weight=loss_weight)

        self.register_buffer("selection_states", torch.zeros(num_experts), persistent=False)
        self.register_buffer("expert_scores_sum", torch.zeros(num_experts), persistent=False)
        self.register_buffer("states_step_count", torch.zeros(1), persistent=False)

    def forward(self, x):
        logits = self.router(x)

        if self.training and torch.is_grad_enabled():
            # 1. 注入噪声：保留一定的探索性，但降低强度，让特征学习占主导
            # noise = torch.randn_like(logits) * 0.1
            # noisy_logits = logits + noise

            # # 2. 选 Top-K
            # _, topk_indices = torch.topk(noisy_logits, self.top_k, dim=1)

            # if self.top_k == 1:
            #     # Top-1 的终极解法
            #     gumbel_weights = F.gumbel_softmax(logits, tau=1.0, hard=True, dim=1)

            #     # gumbel_weights 已经是 one-hot 格式，我们只需要找出是哪个专家即可
            #     _, topk_indices = torch.topk(gumbel_weights, 1, dim=1)

            #     # 因为是 hard=True，选出来的权重绝对是 1.0
            #     selected_weights = torch.gather(gumbel_weights, 1, topk_indices)

            # else:
            #     # 如果未来你要测 Top-K > 1，保留原始 Softmax 即可
            #     global_probs = F.softmax(logits, dim=1)
            #     _, topk_indices = torch.topk(logits + torch.randn_like(logits)*0.1, self.top_k, dim=1)
            #     selected_probs = torch.gather(global_probs, 1, topk_indices)
            #     selected_weights = selected_probs / (selected_probs.sum(dim=1, keepdim=True) + 1e-6)



            # 4. 软路由 (Soft Routing) - 修复梯度与特征衰减
            # 计算全局概率以保留梯度
            # global_probs = F.softmax(logits, dim=1)
            # selected_probs = torch.gather(global_probs, 1, topk_indices)
            # selected_weights = selected_probs

            # 【关键修复】：重归一化！确保选出的 K 个权重和为 1
            # 无论 TopK 是 1, 2 还是 3，输出特征的总体幅值都不会衰减
            # selected_weights = selected_probs / (selected_probs.sum(dim=1, keepdim=True) + 1e-6)
            # if self.top_k == 1:
            #     # 【终极修复：STE (Straight-Through Estimator)】
            #     # 巧妙的数学 trick：
            #     # 前向传播时：selected_probs + 1.0 - selected_probs = 1.0 (特征不衰减)
            #     # 反向传播时：.detach() 的部分没有梯度，梯度 100% 传给 selected_probs (恢复特征学习)
            #     selected_weights = selected_probs + (1.0 - selected_probs).detach()
            # else:
            #     # TopK > 1 时，因为 sum 包含多个不同的概率，P / sum(P) 是有正常梯度的
            #     selected_weights = selected_probs / (selected_probs.sum(dim=1, keepdim=True) + 1e-6)

            global_probs = F.softmax(logits, dim=1)

            if self.top_k == 1:
                # 1. 正常算 Gumbel 选出的 index
                gumbel_weights = F.gumbel_softmax(logits, tau=1.0, hard=True, dim=1)
                _, topk_indices = torch.topk(gumbel_weights, 1, dim=1)

                # 2. 🌟 杀手锏：Epsilon-Greedy 强制探索 (10% 概率)
                # 这保证了没有任何专家会跌到 0% 饿死
                epsilon = 0.10
                # 生成一个随机 Mask，决定哪些样本被强行随机分配
                mask = (torch.rand_like(topk_indices.float()) < epsilon)
                # 生成完全随机的专家索引
                random_indices = torch.randint(0, self.num_experts, topk_indices.shape, device=topk_indices.device)
                # 将 10% 的样本强行替换为随机索引
                topk_indices = torch.where(mask, random_indices, topk_indices)

                # 3. STE 获取对应权重 (依然保持完美的 1.0 特征和流畅的梯度)
                selected_probs = torch.gather(global_probs, 1, topk_indices)
                selected_weights = selected_probs + (1.0 - selected_probs).detach()
            else:
                # TopK > 1 保持不变
                _, topk_indices = torch.topk(logits + torch.randn_like(logits)*0.1, self.top_k, dim=1)
                selected_probs = torch.gather(global_probs, 1, topk_indices)
                selected_weights = selected_probs / (selected_probs.sum(dim=1, keepdim=True) + 1e-6)

            # 3. 数据监测 (No Grad)
            with torch.no_grad():
                flat_indices = topk_indices.flatten()
                counts = torch.bincount(flat_indices, minlength=self.num_experts)
                self.selection_states += counts
                self.expert_scores_sum += logits.mean(dim=0)
                self.states_step_count += 1

            # 5. 负载均衡
            # aux_loss = self.balance_loss_fn(logits, topk_indices)
            aux_loss = torch.tensor(0.0, device=logits.device)  # 目前先关闭负载均衡损失，等后续版本再完善
            MoEAuxCollector.add(aux_loss)

            return selected_weights, topk_indices, logits

        # 推理阶段 (Inference)
        else:
            if self.top_k == 1:
                # 极致极速模式：TopK=1 时，权重恒为 1.0，直接跳过所有 Softmax 和除法运算
                _, topk_indices = torch.topk(logits, 1, dim=1)
                selected_weights = torch.ones_like(topk_indices, dtype=logits.dtype, device=logits.device)
            else:
                # 精度保真模式：TopK > 1 时，必须保留真实的置信度比例！
                # 否则会产生训练(动态权重)与推理(静态均分)的分布偏移
                _, topk_indices = torch.topk(logits, self.top_k, dim=1)

                # 必须像训练时一样计算 Softmax 并重归一化
                global_probs = F.softmax(logits, dim=1)
                selected_probs = torch.gather(global_probs, 1, topk_indices)
                selected_weights = selected_probs / (selected_probs.sum(dim=1, keepdim=True) + 1e-6)

            return selected_weights, topk_indices, logits