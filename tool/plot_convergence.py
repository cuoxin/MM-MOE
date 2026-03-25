import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 核心数据配置 (支持显式指定“最佳模型”)
# ==========================================
# 格式:
# "你想在图例上显示的名字": {"path": "csv真实路径", "is_best": True/False}
MODELS_CONFIG = {
    "Baseline (YOLO12n)": {
        "path": r'/root/autodl-tmp/MM-MOE/runs/myDualData/myDualData-yolo11n-baseline-test-03132/results.csv',
        "is_best": False
    },
    "V16 (Dual-Stream)": {
        "path": r'/root/autodl-tmp/MM-MOE/runs/myDualData/myDualData-yolo11n-MMMOE-backboneV1_16-test-e300-topk1-0114-4/results.csv',
        "is_best": False
    },
    "V19 (No Pass-Through)": {
        "path": r'/root/autodl-tmp/MM-MOE/runs/myDualData/myDualData-yolo11n-MMMOE-backboneV1_19-test-e300-topk1-0117-/results.csv',
        "is_best": False
    },
    # 🌟 将你认为最好的那个模型标记为 is_best: True 🌟
    "V20 (Ours w/ Pass-Through)": {
        "path": r'/root/autodl-tmp/MM-MOE/runs/myDualData/myDualData-yolo11n-MMMOE-backboneV1_20-test-e300-topk1-0117-/results.csv',
        "is_best": False
    }
}

# 预设的学术冷色调备用颜色池 (避免喧宾夺主)
COLORS = ['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2']

# ==========================================
# 2. 数据读取与清洗函数
# ==========================================
def load_and_clean_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        # 去除表头可能含有的不可见空格
        df.columns = df.columns.str.strip()

        epochs = df['epoch'].values
        map_95 = df['metrics/mAP50-95(B)'].values
        # 融合三种 Loss (也可以只取 box_loss 等单项)
        train_loss = df['train/box_loss'].values + df['train/cls_loss'].values + df['train/dfl_loss'].values

        return epochs, map_95, train_loss
    except Exception as e:
        print(f"⚠️ 读取跳过: 无法解析 {file_path} \n错误: {e}")
        return None, None, None

# ==========================================
# 3. 主绘图逻辑
# ==========================================
def plot_multi_convergence():
    print("📊 正在解析所有模型的训练日志...")

    # 设置学术字体和画板
    plt.rcParams['font.family'] = 'serif'
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    ax1, ax2 = axes[0], axes[1]

    color_idx = 0
    max_epoch_plotted = 0

    for model_name, config in MODELS_CONFIG.items():
        csv_path = config["path"]
        is_best = config["is_best"]

        epochs, map_95, train_loss = load_and_clean_csv(csv_path)

        if epochs is None:
            continue

        max_epoch_plotted = max(max_epoch_plotted, max(epochs))

        # 🌟 核心：根据 is_best 变量决定绘图样式 🌟
        if is_best:
            line_color = '#d62728'  # 醒目的学术红
            line_width = 3.0        # 强力加粗
            line_style = '-'        # 坚定的实线
            alpha_val = 1.0         # 完全不透明
            z_order = 10            # 置于图层最顶端，绝对不被遮挡
        else:
            line_color = COLORS[color_idx % len(COLORS)]
            line_width = 1.8        # 偏细
            line_style = '--'       # 对比组用虚线
            alpha_val = 0.75        # 稍微透明，弱化存在感
            z_order = 5             # 放在底层
            color_idx += 1

        # 子图 1: mAP50-95 曲线
        ax1.plot(epochs, map_95, label=model_name, color=line_color,
                 linewidth=line_width, linestyle=line_style, alpha=alpha_val, zorder=z_order)

        # 子图 2: Train Loss 曲线
        ax2.plot(epochs, train_loss, label=model_name, color=line_color,
                 linewidth=line_width, linestyle=line_style, alpha=alpha_val, zorder=z_order)

    # ------------------------------------------
    # 图表细节与美化
    # ------------------------------------------
    # 加一个早期加速收敛的阴影区域 (突出你的收敛优势区)
    early_stage = int(max_epoch_plotted * 0.2) if max_epoch_plotted > 0 else 50

    # 设置子图 1
    ax1.axvspan(0, early_stage, color='gray', alpha=0.08)
    ax1.text(early_stage/2, ax1.get_ylim()[0] + (ax1.get_ylim()[1]-ax1.get_ylim()[0])*0.2,
             "Early Acceleration", ha='center', fontsize=11, color='gray', style='italic')
    ax1.set_title('Validation mAP@50-95 Convergence', fontsize=14, fontweight='bold', pad=10)
    ax1.set_xlabel('Training Epochs', fontsize=12)
    ax1.set_ylabel('mAP@50-95', fontsize=12)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='lower right', fontsize=11, framealpha=0.9)

    # 设置子图 2
    ax2.set_title('Total Training Loss Convergence', fontsize=14, fontweight='bold', pad=10)
    ax2.set_xlabel('Training Epochs', fontsize=12)
    ax2.set_ylabel('Train Loss (Box + Cls + DFL)', fontsize=12)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right', fontsize=11, framealpha=0.9)

    plt.tight_layout()
    save_path = "convergence_multi_models.png"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✅ 完美！多模型收敛曲线对比图已保存至: {save_path}")
    plt.show()

if __name__ == '__main__':
    plot_multi_convergence()