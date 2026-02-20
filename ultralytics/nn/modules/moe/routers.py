import torch
import torch.nn as nn
import torch.nn.functional as F

# 确保这三个文件在你对应的目录下，并且能够被正确引用
from .stats import MoEStatsRecorder
from .loss import LoadBalancingLoss
from .collector import MoEAuxCollector

class UltraEfficientRouter(nn.Module):
    """
    融合版高效路由器：
    1. 架构：采用 YOLO-Master 的 Depthwise Separable Conv 减少参数量。
    2. 策略：
       - 训练时：注入噪声 + Softmax权重 + 计算负载均衡Loss + 统计专家使用率。
       - 推理时：无噪声 + 权重置1 (Hard Routing) + 跳过Loss计算 -> 极致速度。
    """
    def __init__(self, in_channels, num_experts, top_k=1, reduction=16, loss_weight=0.5, Layer_id='MoE_Router'):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.Layer_id = Layer_id

        # --- 1. 高效路由核心网络 (YOLO-Master 风格) ---
        # 激进的通道压缩，但至少保留 4 个通道
        reduced_channels = max(in_channels // reduction, 4)

        self.router = nn.Sequential(
            # 深度卷积 (DW-Conv): 获取空间上下文，大幅减少 FLOPs
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),

            # 逐点卷积 (PW-Conv): 降维
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),

            # 全局池化 (GAP): 变成向量 [B, C_red, 1, 1]
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),

            # 最终分类器: [B, Num_Experts]
            nn.Linear(reduced_channels, num_experts)
        )

        # --- 2. 辅助组件 (来自你的代码) ---
        # 负载均衡损失 (建议权重设为 2.0 或更高，防止坍缩)
        self.balance_loss_fn = LoadBalancingLoss(num_experts, loss_weight)

        # 监测数据 Buffer (不保存到 state_dict，用于训练监控)
        self.register_buffer("selection_states", torch.zeros(num_experts), persistent=False)
        self.register_buffer("expert_scores_sum", torch.zeros(num_experts), persistent=False)
        self.register_buffer("states_step_count", torch.zeros(1), persistent=False)

    def forward(self, x):
        # x: [B, C, H, W]
        # 计算原始 Logits: [B, Num_Experts]
        logits = self.router(x)
        # print(f"👉 [Debug 1] logits 初始版本: {logits._version}")

        # ================== 训练阶段 (Training) ==================
        if self.training and torch.is_grad_enabled():
            safe_logits = logits
            # 1. 注入噪声 (关键：打破对称性，防止死专家)
            # 使用 2.0 的噪声强度（参考你的代码）
            # noise = torch.randn_like(safe_logits) * 0.1
            noisy_logits = safe_logits
            # print(f"👉 [Debug 2] 加噪声后 logits 版本: {logits._version}")

            # 2. 选 Top-K
            # topk_vals: [B, K], topk_indices: [B, K]
            topk_vals, topk_indices = torch.topk(noisy_logits, self.top_k, dim=1)

            # 3. 数据监测 (No Grad)
            with torch.no_grad():
                flat_indices = topk_indices.flatten()
                # 统计每个专家被选中的次数
                counts = torch.bincount(flat_indices, minlength=self.num_experts)
                self.selection_states += counts
                # 统计原始分数的均值
                self.expert_scores_sum += logits.mean(dim=0)
                self.states_step_count += 1

            # 4. 软路由 (Soft Routing) - 保留梯度
            # 注意：要用原始 logits (无噪声) 的对应位置来计算 Softmax，以便梯度回传给 Router
            # raw_topk_logits = torch.gather(logits, 1, topk_indices)
            # selected_weights = F.softmax(raw_topk_logits, dim=1)

            global_probs = F.softmax(logits, dim=1) # 对 4 个专家算 Softmax
            selected_weights = torch.gather(global_probs, 1, topk_indices)

            # 5. 计算负载均衡损失并收集
            aux_loss = self.balance_loss_fn(logits, topk_indices)
            MoEAuxCollector.add(aux_loss)

            # print(f"👉 [Debug 3] 返回前 logits 版本: {logits._version}, safe_logits 版本: {safe_logits._version}")

            return selected_weights, topk_indices, logits

        # ================== 推理阶段 (Inference) ==================
        else:
            # 1. 直接选 Top-K (无噪声)
            _, topk_indices = torch.topk(logits, self.top_k, dim=1)

            # 2. 硬路由 (Hard Routing) - 极致提速
            # 推理时不需要 Softmax 的计算开销，也不需要加权混合
            # 直接把权重置为 1.0，完全依赖专家的输出
            # 形状要匹配 [B, TopK]
            selected_weights = torch.ones_like(topk_indices, dtype=logits.dtype, device=logits.device)

            # 推理时不计算 Aux Loss

            return selected_weights, topk_indices, logits