import torch
import torch.nn as nn
import torch.nn.functional as F

from .loss import LoadBalancingLoss
from .collector import MoEAuxCollector

class UltraEfficientRouter(nn.Module):
    """
    放弃空间特征，直接提取全局上下文，速度达到物理极限。
    """
    def __init__(self, in_channels, num_routed_experts=3, top_k=1, pass_through_expert_nums=1, loss_weight=0.005, noise_multiplier=0.1, Layer_id='MoE_Router', decay_steps=772, reduction=8, routing_weight_mode='consistent'):
        super().__init__()
        self.top_k = top_k
        self.num_routed_experts = num_routed_experts
        self.Layer_id = Layer_id
        self.decay_steps = decay_steps
        self.routing_weight_mode = routing_weight_mode
        self.noise_multiplier = noise_multiplier

        # 直接全局池化，干掉空间维度计算
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        reduced_channels = max(in_channels // reduction, 8)

        self.router = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False),
            nn.GroupNorm(1, reduced_channels),  # 免疫 B=1 的崩溃，且更适合多模态融合
            nn.SiLU(inplace=True),
            nn.Conv2d(reduced_channels, num_routed_experts, kernel_size=1, bias=False),
            nn.GroupNorm(1, num_routed_experts) # 同上
        )

        self.bal_loss_fn = LoadBalancingLoss(
            num_experts=num_routed_experts,
            top_k=top_k,
            loss_weight=loss_weight,
            bg_num=pass_through_expert_nums  # 动态匹配
        )

        self.register_buffer("selection_states", torch.zeros(num_routed_experts), persistent=False)
        self.register_buffer("expert_scores_sum", torch.zeros(num_routed_experts), persistent=False)
        self.register_buffer("states_step_count", torch.zeros(1), persistent=False)
        # 仅用于按 epoch 打印统计，避免影响退火步数 states_step_count。
        self.register_buffer("epoch_states_step_count", torch.zeros(1), persistent=False)

    def forward(self, x):
        B = x.shape[0]
        x_in = self.global_pool(x)
        out = self.router(x_in)
        logits = out.view(B, self.num_routed_experts) # [B, E]
        return self._process_logits(logits)

    def _compute_selected_weights_legacy(self, logits, topk_indices, is_training):
        """保留你原本的权重计算逻辑。"""
        if is_training:
            global_probs = F.softmax(logits, dim=1)
            return torch.gather(global_probs, 1, topk_indices)

        if self.top_k == 1:
            return torch.ones_like(topk_indices, dtype=logits.dtype)

        global_probs = F.softmax(logits, dim=1)
        selected_probs = torch.gather(global_probs, 1, topk_indices)
        return selected_probs / (selected_probs.sum(dim=1, keepdim=True) + 1e-6)

    def _compute_selected_weights_consistent(self, logits, topk_indices):
        """训练/推理一致策略：Top-1 前向恒为 1，Top-K>1 归一化。"""
        global_probs = F.softmax(logits, dim=1)
        selected_probs = torch.gather(global_probs, 1, topk_indices)

        if self.top_k == 1:
            # STE: 前向与推理一致(=1)，梯度仍回传到 selected_probs
            return selected_probs + (1.0 - selected_probs).detach()

        return selected_probs / (selected_probs.sum(dim=1, keepdim=True) + 1e-6)

    def _process_logits(self, logits):
        if self.training and torch.is_grad_enabled():
            current_step = self.states_step_count.item()
            noise_scale = max(0.0, self.noise_multiplier * (1.0 - (current_step / self.decay_steps)))

            noisy_logits = logits + torch.randn_like(logits) * noise_scale
            _, topk_indices = torch.topk(noisy_logits, self.top_k, dim=1)

            if self.routing_weight_mode == 'consistent':
                selected_weights = self._compute_selected_weights_consistent(logits, topk_indices)
            else:
                selected_weights = self._compute_selected_weights_legacy(logits, topk_indices, is_training=True)


            aux_loss = self.bal_loss_fn(logits, topk_indices)
            MoEAuxCollector.add(aux_loss)
            with torch.no_grad():
                self.selection_states += torch.bincount(topk_indices.flatten(), minlength=self.num_routed_experts)
                self.expert_scores_sum += logits.mean(dim=0)
                self.states_step_count += 1
                self.epoch_states_step_count += 1
            return selected_weights, topk_indices
        else:
            if self.top_k == 1:
                topk_indices = torch.argmax(logits, dim=1, keepdim=True)
            else:
                _, topk_indices = torch.topk(logits, self.top_k, dim=1)
            if self.routing_weight_mode == 'consistent':
                selected_weights = self._compute_selected_weights_consistent(logits, topk_indices)
            else:
                selected_weights = self._compute_selected_weights_legacy(logits, topk_indices, is_training=False)
            return selected_weights, topk_indices


# class BatchRouter(nn.Module):
#     # decay_steps 控制噪声退火的时长。
#     # 大约20个epoch
#     def __init__(self, in_channels, num_routed_experts=3, top_k=1, pass_through_expert_nums=1, loss_weight=0.000, Layer_id='MoE_Router', decay_steps=772, reduction=8, pool_scale=8):
#         super().__init__()
#         self.top_k = top_k
#         self.num_routed_experts = num_routed_experts
#         self.Layer_id = Layer_id
#         self.decay_steps = decay_steps

#         # 极简路由网络
#         self.pool_scale = pool_scale
#         reduced_channels = max(in_channels // reduction, 8)
#         self.router = nn.Sequential(
#             # 1. 空间特征提取 (保留局部上下文)
#             nn.Conv2d(in_channels, reduced_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(reduced_channels),
#             nn.SiLU(inplace=True),
#             # 2. 映射到专家维度
#             nn.Conv2d(reduced_channels, num_routed_experts, kernel_size=1, bias=False),
#             # 3. 极度关键的 BN：稳定 Logits，防止某个专家初始得分爆炸导致开局坍塌
#             nn.BatchNorm2d(num_routed_experts)
#         )

#         self.bal_loss_fn = LoadBalancingLoss(
#             num_experts=num_routed_experts,
#             top_k=top_k,
#             loss_weight=loss_weight,
#             bg_num=pass_through_expert_nums,
#             z_loss_weight=0.0 if loss_weight == 0.0 else 1e-3  # 只有当负载均衡损失启用时才使用 Z-Loss
#         )

#         # 监控变量
#         self.register_buffer("selection_states", torch.zeros(num_routed_experts), persistent=False)
#         self.register_buffer("expert_scores_sum", torch.zeros(num_routed_experts), persistent=False)
#         # self.register_buffer("states_step_count", torch.zeros(1), persistent=False)
#         self.current_step = 0

#     def forward(self, x):
#         B, C, H, W = x.shape

#         # yolo_master 的暴力前置池化，极其省算力
#         if H > self.pool_scale and W > self.pool_scale:
#             x_in = F.avg_pool2d(x, kernel_size=self.pool_scale, stride=self.pool_scale)
#         else:
#             x_in = x

#         out = self.router(x_in)  # 输出 shape: [B, Experts, H', W']
#         logits = torch.mean(out, dim=[2, 3])  # 输出 shape: [B, Experts]

#         # ================== 训练阶段 ==================
#         if self.training and torch.is_grad_enabled():

#             # 🌟 创新点 1：线性噪声退火 (前期探索，后期利用)
#             # current_step = self.states_step_count.item()
#             # print(f"Step {current_step}: Router {self.Layer_id} logits: {logits.mean().item():.4f}")
#             noise_scale = max(0.0, 0.1 * (1.0 - (self.current_step / self.decay_steps)))

#             # 将退火噪声加入 Logits 仅用于“决策选谁”
#             noisy_logits = logits + torch.randn_like(logits) * noise_scale
#             _, topk_indices = torch.topk(noisy_logits, self.top_k, dim=1)

#             # 获取真实的概率权重用于“梯度回传”（保证梯度高保真）
#             global_probs = F.softmax(logits, dim=1)
#             selected_weights = torch.gather(global_probs, 1, topk_indices)

#             aux_loss = self.bal_loss_fn(logits, topk_indices)
#             MoEAuxCollector.add(aux_loss)

#             # 记录监控状态
#             with torch.no_grad():
#                 counts = torch.bincount(topk_indices.flatten(), minlength=self.num_routed_experts)
#                 self.selection_states += counts
#                 self.expert_scores_sum += logits.mean(dim=0)
#                 self.current_step += 1

#             return selected_weights, topk_indices

#         # ================== 推理阶段 ==================
#         else:
#             if self.top_k == 1:
#                 # 🚀 提速细节 4：当 k=1 时，用极速的 argmax 替代重量级的 topk
#                 topk_indices = torch.argmax(logits, dim=1, keepdim=True)
#                 selected_weights = torch.ones_like(topk_indices, dtype=logits.dtype)
#             else:
#                 # 只有 k>1 时，才迫不得已使用 topk
#                 _, topk_indices = torch.topk(logits, self.top_k, dim=1)
#                 global_probs = F.softmax(logits, dim=1)
#                 selected_probs = torch.gather(global_probs, 1, topk_indices)
#                 selected_weights = selected_probs / (selected_probs.sum(dim=1, keepdim=True) + 1e-6)

#             return selected_weights, topk_indices

# class UltraEfficientRouter(nn.Module):
#     """
#     统一的 MoE 路由入口包装器 (Facade Pattern)。
#     不修改原有的 GroupNormRouter 和 BatchRouter 代码，通过 router_type 参数动态切换。
#     自动处理底层参数不一致的问题。
#     """
#     def __init__(self,
#                  in_channels,
#                  num_routed_experts=3,
#                  top_k=1,
#                  pass_through_expert_nums=1,
#                  loss_weight=0.005,           # 统一的负载均衡损失权重参数
#                  Layer_id='MoE_Router',
#                  decay_steps=772,
#                  reduction=8,
#                  pool_scale=8,                # BatchRouter 独有参数
#                  router_type='group'):        # 核心切换开关：'group' 或 'batch'
#         super().__init__()

#         self.router_type = router_type
#         self.Layer_id = Layer_id

#         if self.router_type == 'group':
#             # 路线 A: 实例化 GroupRouter (带负载均衡)
#             self.router = GroupRouter(
#                 in_channels=in_channels,
#                 num_routed_experts=num_routed_experts,
#                 top_k=top_k,
#                 pass_through_expert_nums=pass_through_expert_nums,
#                 loss_weight=loss_weight,
#                 Layer_id=Layer_id,
#                 decay_steps=decay_steps,
#                 reduction=reduction
#             )

#         elif self.router_type == 'batch':
#             # 路线 B: 实例化 BatchRouter (无负载均衡)
#             # 原始的 BatchRouter 没有 pass_through_expert_nums 参数，这里自动屏蔽
#             self.router = BatchRouter(
#                 in_channels=in_channels,
#                 num_routed_experts=num_routed_experts,
#                 top_k=top_k,
#                 pass_through_expert_nums=pass_through_expert_nums,
#                 loss_weight=0.0,
#                 Layer_id=Layer_id,
#                 decay_steps=decay_steps,
#                 reduction=reduction,
#                 pool_scale=pool_scale
#             )

#         else:
#             raise ValueError(f"❌ 初始化失败: 未知的 router_type '{router_type}'。仅支持 'group' 或 'batch'。")

#     def forward(self, x):
#         """
#         前向传播直接透传给底层的路由器，完全保留原始类的内部逻辑。
#         """
#         return self.router(x)