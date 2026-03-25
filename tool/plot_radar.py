import warnings
warnings.filterwarnings('ignore')
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ==========================================
# 1. 核心评估配置区域
# ==========================================
MODELS_TO_TEST = {
    "Baseline (YOLO11n)": r'/root/autodl-tmp/MM-MOE/runs/myDualData/myDualData-yolo11n-baseline-test-03132/weights/best.pt',
    "V20 Ours (MM-MoE)":  r'/root/autodl-tmp/MM-MOE/runs/myDualData/myDualData-yolo11n-MMMOE-backboneV2_22-test-e300-topk1-0119-13/weights/best.pt'
}

BASE_DATASET_DIR = r'/root/autodl-tmp/datasets/sorted_scenesV2_yolo_format'

SCENARIOS = {
    "Normal": os.path.join(BASE_DATASET_DIR, '1_Normal'),
    "Dense": os.path.join(BASE_DATASET_DIR, '2_Dense'),
    "Night": os.path.join(BASE_DATASET_DIR, '3_Night'),
    "Small": os.path.join(BASE_DATASET_DIR, '4_Small'),
    "Sea Fog": os.path.join(BASE_DATASET_DIR, '5_SeaFog'),
    "Strong Light": os.path.join(BASE_DATASET_DIR, '6_StrongLight'),
    "Background": os.path.join(BASE_DATASET_DIR, '7_Background')
}

NUM_CLASSES = 1
CLASS_NAMES = ['ship']

# ==========================================
# 2. 自动化评估与数据收集引擎
# ==========================================
def generate_temp_yaml(scene_root_path, yaml_name="temp_scenario.yaml"):
    val_images_path = os.path.join(scene_root_path, 'images', 'visible', 'val')
    data = {
        'path': scene_root_path,
        'train': val_images_path,
        'val': val_images_path,
        'test': val_images_path,
        'nc': NUM_CLASSES,
        'names': CLASS_NAMES
    }
    with open(yaml_name, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, allow_unicode=True)
    return yaml_name

def run_auto_evaluation():
    # 🌟 更新：分别建立 mAP50 和 mAP50-95 的存储字典
    final_maps50 = {model_name: [] for model_name in MODELS_TO_TEST.keys()}
    final_maps95 = {model_name: [] for model_name in MODELS_TO_TEST.keys()}

    categories = list(SCENARIOS.keys())
    temp_yaml = "temp_scenario.yaml"

    print(f"\n{'='*70}")
    print("🚀 启动自动化端到端多场景评估流水线 (mAP50 & mAP50-95)")
    print(f"{'='*70}")

    for model_name, weights_path in MODELS_TO_TEST.items():
        print(f"\n📦 当前评估模型: {model_name}")
        model = YOLO(weights_path)

        for scene_name, scene_root_path in SCENARIOS.items():
            print(f"  -> 正在评估场景: {scene_name} ...")

            if not os.path.exists(scene_root_path):
                print(f"     ❌ 错误: 找不到场景目录 {scene_root_path}")
                final_maps50[model_name].append(0.0)
                final_maps95[model_name].append(0.0)
                continue

            generate_temp_yaml(scene_root_path, temp_yaml)

            try:
                results = model.val(
                    data=temp_yaml,
                    split='val',
                    imgsz=640,
                    batch=16,
                    use_simotm="RGBRGB6C",
                    channels=6,
                    plots=False,
                    verbose=False
                )

                # 🌟 更新：同时提取两种 mAP 并转为百分比
                map50_val = results.box.map50 * 100
                map95_val = results.box.map * 100

                final_maps50[model_name].append(map50_val)
                final_maps95[model_name].append(map95_val)
                print(f"     ✅ 评估完成: mAP@50 = {map50_val:.1f}% | mAP@50-95 = {map95_val:.1f}%")

            except Exception as e:
                print(f"     ❌ 评估失败: {e}")
                final_maps50[model_name].append(0.0)
                final_maps95[model_name].append(0.0)

    if os.path.exists(temp_yaml):
        os.remove(temp_yaml)

    return categories, final_maps50, final_maps95

# ==========================================
# 3. 高级双子星雷达图绘制模块
# ==========================================
def draw_single_radar(ax, categories, model_results, title):
    """辅助函数：在指定的 ax 上绘制单个雷达图"""
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # 标签设置
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=12, fontweight='bold')
    ax.tick_params(axis='x', pad=15)

    # 动态计算当前图表的 Y 轴自适应范围
    all_values = [val for maps in model_results.values() for val in maps]
    min_val = max(0, int(min(all_values) / 10) * 10 - 10)
    max_val = min(100, int(max(all_values) / 10) * 10 + 10)

    ax.set_ylim(min_val, max_val)
    yticks = np.arange(min_val + 10, max_val, 10)
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(int(y)) for y in yticks], color="grey", size=10)
    ax.set_rlabel_position(30)

    colors = ['#7f7f7f', '#d62728', '#1f77b4', '#2ca02c']
    styles = ['--', '-', '-.', ':']

    for idx, (model_name, map_scores) in enumerate(model_results.items()):
        closed_maps = map_scores + map_scores[:1]

        is_ours = "Ours" in model_name or "MoE" in model_name
        color = '#d62728' if is_ours else colors[idx % len(colors)]
        lw = 3.5 if is_ours else 2.0
        ls = '-' if is_ours else styles[idx % len(styles)]
        alpha_fill = 0.25 if is_ours else 0.1

        # 画线、填充与散点
        ax.plot(angles, closed_maps, linewidth=lw, linestyle=ls, color=color, label=model_name)
        ax.fill(angles, closed_maps, color=color, alpha=alpha_fill)
        ax.scatter(angles[:-1], map_scores, color=color, s=40 if is_ours else 20, zorder=10)

    ax.set_title(title, size=15, fontweight='bold', pad=25)


def plot_academic_dual_radar(categories, results_map50, results_map95):
    print("\n🎨 正在生成双子星雷达对比图 (mAP50 & mAP50-95)...")

    plt.rcParams['font.family'] = 'serif'
    # 创建 1行2列 的极坐标图画板，横向加宽以容纳两张图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), subplot_kw=dict(polar=True), dpi=300)

    # 绘制左图 (mAP50)
    draw_single_radar(ax1, categories, results_map50, '(a) Robustness in mAP@50')

    # 绘制右图 (mAP50-95)
    draw_single_radar(ax2, categories, results_map95, '(b) Robustness in mAP@50-95')

    # 提取图例，放置在整个画面的正中上方
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05),
               ncol=len(labels), fontsize=13, frameon=True, title="Architecture Variants", title_fontsize=14)

    plt.tight_layout()
    # 稍微调整两张图的间距，防止标签重叠
    plt.subplots_adjust(wspace=0.3, top=0.85)

    save_path = 'dual_radar_chart_auto_eval.png'
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✅ 完美！双子星雷达图已保存至: {save_path}")
    plt.show()

if __name__ == '__main__':
    # 1. 执行自动跑库拿数据
    cats, results_50, results_95 = run_auto_evaluation()

    # 打印 mAP50 矩阵
    print("\n📊 自动收集数据 [mAP@50]:")
    for m, vals in results_50.items():
        val_str = ", ".join([f"{v:.1f}%" for v in vals])
        print(f"[{m}]: {val_str}")

    # 打印 mAP50-95 矩阵
    print("\n📊 自动收集数据 [mAP@50-95]:")
    for m, vals in results_95.items():
        val_str = ", ".join([f"{v:.1f}%" for v in vals])
        print(f"[{m}]: {val_str}")

    # 2. 画双子图
    plot_academic_dual_radar(cats, results_50, results_95)