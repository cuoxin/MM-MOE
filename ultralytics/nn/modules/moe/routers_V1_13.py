import torch
import torch.nn as nn
import torch.nn.functional as F

class UltraEfficientRouter(nn.Module):
    # decay_steps 控制噪声退火的时长。
    # 大约20个epoch
    def __init__(self, in_channels, num_routed_experts=3, top_k=1, Layer_id='MoE_Router', decay_steps=1930):
        super().__init__()
        self.top_k = top_k
        self.num_routed_experts = num_routed_experts
        self.Layer_id = Layer_id
        self.decay_steps = decay_steps

        # 极简路由网络
        reduced_channels = max(in_channels // 16, 4)
        self.router = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 2, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, reduced_channels, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(reduced_channels, num_routed_experts)
        )

        # 监控变量
        self.register_buffer("selection_states", torch.zeros(num_routed_experts), persistent=False)
        self.register_buffer("expert_scores_sum", torch.zeros(num_routed_experts), persistent=False)
        self.register_buffer("states_step_count", torch.zeros(1), persistent=False)

    def forward(self, x):
        logits = self.router(x)

        # ================== 训练阶段 ==================
        if self.training and torch.is_grad_enabled():

            # 🌟 创新点 1：线性噪声退火 (前期探索，后期利用)
            current_step = self.states_step_count.item()
            # print(f"Step {current_step}: Router {self.Layer_id} logits: {logits.mean().item():.4f}")
            noise_scale = max(0.0, 0.5 * (1.0 - (current_step / self.decay_steps)))

            # 将退火噪声加入 Logits 仅用于“决策选谁”
            noisy_logits = logits + torch.randn_like(logits) * noise_scale
            _, topk_indices = torch.topk(noisy_logits, self.top_k, dim=1)

            # 获取真实的概率权重用于“梯度回传”（保证梯度高保真）
            global_probs = F.softmax(logits, dim=1)
            selected_weights = torch.gather(global_probs, 1, topk_indices)

            # 记录监控状态
            with torch.no_grad():
                counts = torch.bincount(topk_indices.flatten(), minlength=self.num_routed_experts)
                self.selection_states += counts
                self.expert_scores_sum += logits.mean(dim=0)
                self.states_step_count += 1

            return selected_weights, topk_indices

        # ================== 推理阶段 ==================
        else:
            # 零噪声，绝对特征驱动。TopK=1 时满特征传递，速度拉满。
            _, topk_indices = torch.topk(logits, self.top_k, dim=1)
            if self.top_k == 1:
                selected_weights = torch.ones_like(topk_indices, dtype=logits.dtype, device=logits.device)
            else:
                global_probs = F.softmax(logits, dim=1)
                selected_probs = torch.gather(global_probs, 1, topk_indices)
                selected_weights = selected_probs / (selected_probs.sum(dim=1, keepdim=True) + 1e-6)

            return selected_weights, topk_indices