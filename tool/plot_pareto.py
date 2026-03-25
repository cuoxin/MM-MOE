import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 核心数据配置 (请填入你真实的实验数据)
# ==========================================
# 数据格式: { "模型名称": (推理延迟ms, mAP50-95, 参数量M) }
# 注意：参数量(M)用于控制气泡的大小，差异越大视觉效果越好
model_data = {
    "Baseline (YOLOv11)": (4.6, 0.705, 3.2),
    "V16 (Dual-Stream)":  (5.2, 0.720, 6.1), # 双流网络通常精度高但变慢、变大
    "V17 (Standard MoE)": (4.8, 0.723, 7.5), # 标准MoE参数激增，速度略微下降
    "V19 (Neck Fusion)":  (4.2, 0.731, 6.8), # 优化后速度提升，精度提升
    "V20 (Ours)":         (3.7, 0.742, 6.8)  # 终极形态：加入直通专家，参数不变，速度起飞，精度最高
}

# 颜色配置 (突出 Ours)
colors = {
    "Baseline (YOLOv11)": "#7f7f7f",  # 灰色
    "V16 (Dual-Stream)":  "#1f77b4",  # 蓝色
    "V17 (Standard MoE)": "#2ca02c",  # 绿色
    "V19 (Neck Fusion)":  "#ff7f0e",  # 橙色
    "V20 (Ours)":         "#d62728"   # 红色 (视觉中心)
}

# ==========================================
# 2. 数据解析与帕累托前沿计算
# ==========================================
names = list(model_data.keys())
latencies = np.array([model_data[name][0] for name in names])
maps = np.array([model_data[name][1] * 100 for name in names]) # 转换为百分比 %
params = np.array([model_data[name][2] for name in names])

# 气泡大小缩放系数 (根据你的参数量大小微调这个乘数)
bubble_sizes = params * 100

# 计算帕累托前沿 (Latency 越小越好，mAP 越大越好)
# 1. 按 Latency 从小到大排序
sorted_indices = np.argsort(latencies)
sorted_latencies = latencies[sorted_indices]
sorted_maps = maps[sorted_indices]

pareto_latencies = []
pareto_maps = []
current_max_map = -1.0

# 2. 只有当 mAP 比之前所有更快模型都要高时，才属于帕累托前沿
for lat, m in zip(sorted_latencies, sorted_maps):
    if m > current_max_map:
        pareto_latencies.append(lat)
        pareto_maps.append(m)
        current_max_map = m

# ==========================================
# 3. 开始绘图
# ==========================================
# 设置全局字体和画板样式
plt.rcParams['font.family'] = 'serif'
fig, ax = plt.subplots(figsize=(10, 7), dpi=300)

# 绘制帕累托前沿线 (虚线)
ax.plot(pareto_latencies, pareto_maps, linestyle='--', color='gray', linewidth=2, alpha=0.6, zorder=1, label='Pareto Frontier')

# 绘制气泡散点
for i, name in enumerate(names):
    # Ours 模型加粗，边框特殊处理
    is_ours = "Ours" in name
    edge_color = 'black' if is_ours else 'white'
    linewidth = 2.5 if is_ours else 1.0
    alpha_val = 0.9 if is_ours else 0.75

    scatter = ax.scatter(latencies[i], maps[i], s=bubble_sizes[i], c=colors[name],
                         alpha=alpha_val, edgecolors=edge_color, linewidth=linewidth, zorder=2)

    # 添加文本标签 (Ours 放上面，其他的微调位置避免遮挡)
    text_y_offset = 0.3 if is_ours else -0.5
    font_weight = 'bold' if is_ours else 'normal'
    font_size = 13 if is_ours else 11

    ax.text(latencies[i], maps[i] + text_y_offset, name,
            fontsize=font_size, fontweight=font_weight, ha='center', va='bottom', zorder=3)

# ==========================================
# 4. 图表美化与细节调整
# ==========================================
ax.set_title("Performance vs. Latency Trade-off", fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel("Inference Latency (ms) $\leftarrow$ Faster", fontsize=14)
ax.set_ylabel("mAP@50-95 (%)", fontsize=14)

# 添加网格线
ax.grid(True, linestyle=':', alpha=0.7, zorder=0)

# 添加图例说明 (气泡大小含义)
# 制造几个假的散点只为了画图例
scatter_legend = [ax.scatter([], [], s=size*100, c='gray', alpha=0.5, edgecolors='white') for size in [3, 5, 7]]
legend1 = ax.legend(scatter_legend, ['3M', '5M', '7M'], title="Params", loc='lower right', frameon=True, fontsize=11)
ax.add_artist(legend1) # 保持气泡图例

# 添加帕累托前沿图例
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc='upper left', fontsize=12)

# 优化坐标轴边距
plt.margins(x=0.15, y=0.15)

# 保存与展示
save_path = "pareto_frontier_bubble.png"
plt.tight_layout()
plt.savefig(save_path, bbox_inches='tight')
print(f"✅ 帕累托气泡图已保存至: {save_path}")
plt.show()