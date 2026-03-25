'''
动态延迟分布小提琴图脚本

'''

import warnings
warnings.filterwarnings('ignore')
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import torch
import numpy as np
import pandas as pd
import cv2
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO

# ==========================================
# 1. 核心配置区域
# ==========================================
WEIGHTS_BASELINE = r'/root/autodl-tmp/MM-MOE/runs/myDualData/myDualData-yolo11n-baseline-test-03132/weights/best.pt'
WEIGHTS_V20 = r'/root/autodl-tmp/MM-MOE/runs/myDualData/myDualData-yolo11n-MMMOE-backboneV2_22-test-e300-topk1-0119-13/weights/best.pt'

# 你的场景数据集路径 (复用你之前的配置)
SCENARIOS = {
    "Normal": {
        "rgb": r'/root/autodl-tmp/datasets/sorted_scenes/1_Normal/visible',
        "ir":  r'/root/autodl-tmp/datasets/sorted_scenes/1_Normal/infrared'
    },
    "Night": {
        "rgb": r'/root/autodl-tmp/datasets/sorted_scenes/2_Night/visible',
        "ir":  r'/root/autodl-tmp/datasets/sorted_scenes/2_Night/infrared'
    },
    "Fog": {
        "rgb": r'/root/autodl-tmp/datasets/sorted_scenes/3_SeaFog/visible',
        "ir":  r'/root/autodl-tmp/datasets/sorted_scenes/3_SeaFog/infrared'
    },
    "StrongLight": {
        "rgb": r'/root/autodl-tmp/datasets/sorted_scenes/4_StrongLight/visible',
        "ir":  r'/root/autodl-tmp/datasets/sorted_scenes/4_StrongLight/infrared'
    },
    "Background": {
        "rgb": r'/root/autodl-tmp/datasets/sorted_scenes/5_Background/visible',
        "ir":  r'/root/autodl-tmp/datasets/sorted_scenes/5_Background/infrared'
    }
}

# ==========================================
# 2. 图像加载与 6 通道拼接
# ==========================================
def load_6ch_tensor(img_path_rgb, img_path_ir, device, imgsz=(640, 640)):
    img1 = cv2.imread(img_path_rgb)
    img2 = cv2.imread(img_path_ir)
    if img1 is None or img2 is None:
        return None
    img1 = cv2.cvtColor(cv2.resize(img1, imgsz), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.resize(img2, imgsz), cv2.COLOR_BGR2RGB)
    t1 = torch.from_numpy(img1).permute(2, 0, 1).float() / 255.0
    t2 = torch.from_numpy(img2).permute(2, 0, 1).float() / 255.0
    return torch.cat((t1, t2), dim=0).unsqueeze(0).to(device)

# ==========================================
# 3. 硬件级高精度延迟测试核心逻辑
# ==========================================
def measure_latency(model, tensor_6ch):
    """使用 CUDA Event 进行严格的同步测速"""
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        # GPU 预热 (极其重要，排除启动开销)
        for _ in range(3):
            _ = model(tensor_6ch)

        torch.cuda.synchronize() # 等待之前的所有操作完成
        starter.record()         # 记录开始时间

        _ = model(tensor_6ch)    # 前向推理

        ender.record()           # 记录结束时间
        torch.cuda.synchronize() # 强制等待 GPU 算完

        latency = starter.elapsed_time(ender) # 单位: 毫秒 (ms)
    return latency

# ==========================================
# 4. 主干逻辑：遍历数据集收集耗时
# ==========================================
def collect_real_latency_data():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("🚀 正在加载模型权重...")

    # 我们只测试底层 model 的前向速度，避开 NMS 后处理的干扰，以纯粹展示特征提取的算力差异
    baseline_model = YOLO(WEIGHTS_BASELINE).model.to(device).eval()
    v20_model = YOLO(WEIGHTS_V20).model.to(device).eval()

    results = []

    for scene, paths in SCENARIOS.items():
        print(f"\n📊 开始测试场景: {scene}")
        rgb_images = sorted(glob.glob(os.path.join(paths['rgb'], "*.jpg")) + glob.glob(os.path.join(paths['rgb'], "*.png")))

        for i, rgb_path in enumerate(rgb_images):
            ir_path = os.path.join(paths['ir'], os.path.basename(rgb_path))
            if not os.path.exists(ir_path):
                continue

            tensor_6ch = load_6ch_tensor(rgb_path, ir_path, device)
            if tensor_6ch is None: continue

            # 测 Baseline
            lat_base = measure_latency(baseline_model, tensor_6ch)
            results.append({"Scenario": scene, "Model": "Baseline (Static)", "Latency (ms)": lat_base})

            # 测 V20
            lat_v20 = measure_latency(v20_model, tensor_6ch)
            results.append({"Scenario": scene, "Model": "V20 Ours (Dynamic)", "Latency (ms)": lat_v20})

            if (i+1) % 50 == 0:
                print(f"  已处理 {i+1}/{len(rgb_images)} 张...")

    return pd.DataFrame(results)

# ==========================================
# 5. 绘图逻辑
# ==========================================
def plot_grouped_violin(df):
    print("\n🎨 正在生成优化版的小提琴对比图...")
    plt.rcParams['font.family'] = 'serif'
    fig, ax = plt.subplots(figsize=(13, 7), dpi=300)

    palette = {'Baseline (Static)': '#888888', 'V20 Ours (Dynamic)': '#d62728'}

    # 1. 绘制小提琴图 (🌟 移除 cut=0, 加入 bw_adjust 进行学术平滑)
    import seaborn as sns
    sns.violinplot(
        data=df,
        x='Scenario',
        y='Latency (ms)',
        hue='Model',
        split=True,
        palette=palette,
        inner='box',
        ax=ax,
        linewidth=1.2,
        alpha=0.8,
        bw_adjust=0.8  # 🌟 核心平滑参数：降低对微小系统抖动的敏感度，让提琴形状更丰满
    )

    # 2. 动态截断异常值，放大核心区域
    y_min = df['Latency (ms)'].quantile(0.01) - 1.0
    y_max = df['Latency (ms)'].quantile(0.95) + 2.0
    ax.set_ylim(y_min, y_max)

    # 3. 🌟 文本标注优化：展示绝对时间差，显得更真实 🌟
    scenarios = df['Scenario'].unique()
    for i, scene in enumerate(scenarios):
        scene_data = df[df['Scenario'] == scene]
        base_median = scene_data[scene_data['Model'] == 'Baseline (Static)']['Latency (ms)'].median()
        v20_median = scene_data[scene_data['Model'] == 'V20 Ours (Dynamic)']['Latency (ms)'].median()

        # 计算加速比和绝对差值
        speedup = ((base_median - v20_median) / base_median) * 100
        time_saved = base_median - v20_median

        # 标注文本：例如 "-1.05ms (↓7.2%)"
        ax.text(i, y_max - 0.5, f"-{time_saved:.2f}ms\n(↓{speedup:.1f}%)",
                ha='center', va='top', fontsize=12, fontweight='bold', color='#d62728',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))

    # 图表美化
    ax.set_title('Real-World Inference Latency Distribution Across Scenarios',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Testing Scenario', fontsize=14, fontweight='bold')
    ax.set_ylabel('Per-Image Inference Latency (ms)', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # 调整图例
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, title="Architecture Type", title_fontsize=12, fontsize=11, loc='upper right')

    plt.tight_layout()
    save_path = 'real_latency_scenario_violin_smoothed.png'
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✅ 自然平滑版图表已保存至: {save_path}")
    plt.show()

if __name__ == '__main__':
    df_results = collect_real_latency_data()
    plot_grouped_violin(df_results)