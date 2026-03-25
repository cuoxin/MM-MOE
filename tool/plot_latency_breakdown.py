'''水平堆叠条形图（算力分解/延迟节省图）'''

import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 核心数据配置 (请根据你真实的测速数据微调)
# ==========================================
# 测试场景 (按照直通专家利用率从高到低排序，视觉效果最好)
scenarios = ['Background (纯背景)', 'Sea Fog (海雾)', 'Night (夜间)', 'Normal (常规)', 'Strong Light (强反光)']

# Baseline 在各个场景下的延迟 (由于是静态网络，基本死死卡在一个固定值)
baseline_latency = np.array([4.60, 4.61, 4.59, 4.60, 4.62])

# V20 (Ours) 在各个场景下的真实延迟 (按需计算，越简单的场景延迟越低)
# 这里的数据基于你之前的专家利用率推算：背景全走直通专家，速度最快；强反光走重火力专家，速度最慢
v20_actual_latency = np.array([2.95, 3.30, 3.45, 3.85, 4.10])

# 计算 V20 相比 Baseline 节省下来的时间 (这就是直通专家的物理贡献！)
v20_saved_latency = baseline_latency - v20_actual_latency

# ==========================================
# 2. 坐标与画板初始化
# ==========================================
plt.rcParams['font.family'] = 'serif'
fig, ax = plt.subplots(figsize=(12, 7), dpi=300)

y = np.arange(len(scenarios))  # Y 轴基准位置
height = 0.35                  # 柱子的宽度/高度

# ==========================================
# 3. 核心绘图逻辑：堆叠与对比
# ==========================================
# (1) 绘制 Baseline 柱子 (灰色实心)
bars_base = ax.barh(y + height/2, baseline_latency, height,
                    label='Baseline (Static Dense Compute)', color='#7f7f7f', alpha=0.85)

# (2) 绘制 V20 实际消耗的延迟 (红色实心)
bars_v20_active = ax.barh(y - height/2, v20_actual_latency, height,
                          label='V20 Ours (Active MoE Compute)', color='#d62728', alpha=0.9)

# (3) 🌟 绝杀：绘制 V20 被省下来的延迟 (镂空阴影填充) 🌟
# 注意：它的起点 (left) 是实际延迟的终点，刚好和 Baseline 对齐，形成视觉上的完整对比
bars_v20_saved = ax.barh(y - height/2, v20_saved_latency, height, left=v20_actual_latency,
                         label='Compute Saved by Pass-Through Expert',
                         color='white', edgecolor='#d62728', hatch='///', linewidth=1.5)

# ==========================================
# 4. 图表标注与数据可视化增强
# ==========================================
# 给 V20 的节省区域标上显眼的百分比和绝对数值
for i in range(len(scenarios)):
    saved_ms = v20_saved_latency[i]
    total_ms = baseline_latency[i]
    percentage = (saved_ms / total_ms) * 100

    # 在镂空区域的正中间写字
    text_x = v20_actual_latency[i] + saved_ms / 2
    ax.text(text_x, y[i] - height/2, f"-{percentage:.1f}%\n({saved_ms:.2f}ms)",
            ha='center', va='center', color='#8b0000', fontsize=11, fontweight='bold')

    # 在 Baseline 柱子末尾标上总时间
    ax.text(total_ms + 0.05, y[i] + height/2, f"{total_ms:.2f} ms",
            ha='left', va='center', color='#333333', fontsize=11)

    # 在 V20 实际计算柱子末尾标上实际时间
    ax.text(v20_actual_latency[i] - 0.1, y[i] - height/2, f"{v20_actual_latency[i]:.2f} ms",
            ha='right', va='center', color='white', fontsize=11, fontweight='bold')

# ==========================================
# 5. 图表美化与细节
# ==========================================
ax.set_title('Inference Latency Breakdown Across Scenarios\n(Visualizing Conditional Computation Savings)',
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Per-Image Inference Latency (ms)', fontsize=14, fontweight='bold')
ax.set_yticks(y)
ax.set_yticklabels(scenarios, fontsize=13)
ax.tick_params(axis='x', labelsize=12)

# 限制 X 轴的范围，留出文字显示的空间
ax.set_xlim(0, max(baseline_latency) + 0.6)

# 添加垂直虚线辅助阅读
ax.xaxis.grid(True, linestyle='--', alpha=0.5, color='gray')
ax.set_axisbelow(True) # 让网格线在柱子下方

# 优化图例 (放在图表右下角的空白处)
ax.legend(loc='lower right', fontsize=12, framealpha=0.9, title="Latency Composition", title_fontsize=13)

plt.tight_layout()
save_path = 'latency_breakdown_stacked_bar.png'
plt.savefig(save_path, bbox_inches='tight')
print(f"✅ 算力分解条形图已保存至: {save_path}")
plt.show()