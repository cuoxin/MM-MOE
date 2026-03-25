import re
import matplotlib.pyplot as plt

def parse_and_plot_moe_logs(log_file_path, out_img_path, target_layer="8"):
    """
    解析 MoE 日志并绘制指定层的专家利用率折线图
    :param log_file_path: 日志文本文件路径
    :param out_img_path: 输出图片路径
    :param target_layer: 想要绘制的层级名字（例如 "8", "10", "17", "20"）
    """
    epochs = []
    # 数据结构: data[layer_name][expert_id] = [pct_epoch1, pct_epoch2, ...]
    data = {}

    current_epoch = None

    # 正则表达式匹配规则
    epoch_pattern = re.compile(r"==================== MoE Expert States \(Epoch (\d+)\) ====================")
    layer_pattern = re.compile(r"Layer (.*?)_Router \[Select%\]:\s*(.*)")
    # 匹配诸如 "Exp0:  55.4%" 中的数字
    exp_pattern = re.compile(r"Exp(\d+):\s*([\d.]+)%")

    # 1. 解析日志文件
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()

                # 提取 Epoch
                epoch_match = epoch_pattern.search(line)
                if epoch_match:
                    current_epoch = int(epoch_match.group(1))
                    if current_epoch not in epochs:
                        epochs.append(current_epoch)
                    continue

                # 提取 Layer 和 专家比例
                layer_match = layer_pattern.search(line)
                if layer_match and current_epoch is not None:
                    layer_name = layer_match.group(1)
                    exp_str = layer_match.group(2)

                    if layer_name not in data:
                        data[layer_name] = {}

                    # 解析当前行的所有专家比例
                    experts = exp_pattern.findall(exp_str)
                    for exp_id_str, pct_str in experts:
                        exp_id = int(exp_id_str)
                        pct = float(pct_str)

                        if exp_id not in data[layer_name]:
                            data[layer_name][exp_id] = []

                        data[layer_name][exp_id].append(pct)

    except FileNotFoundError:
        print(f"找不到日志文件: {log_file_path}，请检查路径。")
        return

    # 2. 检查目标层是否存在
    if target_layer not in data:
        print(f"未在日志中找到 Layer {target_layer}。")
        print(f"可用的 Layer 有: {list(data.keys())}")
        return

    # 3. 绘制折线图
    layer_experts = data[target_layer]

    # 设置图形样式（高分辨率，适合论文）
    plt.figure(figsize=(10, 6), dpi=150)

    # 定义一些好看的颜色
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']

    for i, exp_id in enumerate(sorted(layer_experts.keys())):
        pcts = layer_experts[exp_id]

        # 补齐长度（防止日志中断导致维度不匹配）
        if len(pcts) < len(epochs):
            pcts.extend([None] * (len(epochs) - len(pcts)))

        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        # 画线
        plt.plot(epochs, pcts, marker=marker, linewidth=2, markersize=6,
                 color=color, label=f'Expert {exp_id}')

    # 图表细节美化
    plt.title(f'MoE Expert Utilization Trend - Layer {target_layer}', fontsize=16, fontweight='bold')
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Selection Percentage (%)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # 设置Y轴范围 0 - 100，更直观
    plt.ylim(0, 100)

    # 添加网格线和图例
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12, loc='best')

    plt.tight_layout()

    # 保存图片（方便直接插入论文）
    output_filename = out_img_path + "/moe_expert_utilization_layer_{}.png".format(target_layer)
    plt.savefig(output_filename)
    print(f"图表已成功生成并保存为: {output_filename}")

    # 弹出展示窗口
    plt.show()

if __name__ == "__main__":
    # 在这里指定你的日志文本文件名
    LOG_FILE = "/root/autodl-tmp/MM-MOE/runs/myDualData/myDualData-yolo11n-MMMOE-backboneV1_20-test-e300-topk1-0117-/moe_states.txt"
    OUT_IMG = '/root/autodl-tmp/MM-MOE/tool/result_temp'

    # 在这里修改你想查看的层，比如 "8", "10", "17", 或 "20"
    # TARGET_LAYER = "17"

    for i in list(["9", "12", "21", "24"]):
        parse_and_plot_moe_logs(LOG_FILE, OUT_IMG, i)