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
    "Baseline (YOLO11n)": r'/root/autodl-tmp/MM-MOE/runs/final_MMMOE/baseline_P3/weights/best.pt',
    "V6_0 Ours (MM-MoE)": r'/root/autodl-tmp/MM-MOE/runs/all_data_V1_3/myDualData-MMMOE-backbone-V8_1/weights/best.pt'
}

BASE_DATASET_DIR = r'/root/autodl-tmp/datasets/val_sorted_scenes'

NUM_CLASSES = 1
CLASS_NAMES = ['ship']

# 评估阶段的稳定性参数（防止多进程/大 batch 导致进程被系统杀掉）
VAL_IMGSZ = 640
VAL_BATCH = 8
VAL_WORKERS = 0

# 雷达图显示模式:
#   "absolute": 使用真实 mAP 百分比
#   "relative": 每个场景按最佳模型归一化到 100（更突出模型强弱层次）
RADAR_MODE = "relative"
HIDE_NON_IMPROVED_SCENES = False

# ==========================================
# 2. 自动化评估与数据收集引擎
# ==========================================
def discover_scenarios(base_dir):
    """自动发现场景目录，要求每个场景符合 images/visible/val 数据结构。"""
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"找不到数据集根目录: {base_dir}")

    scenario_map = {}
    for name in sorted(os.listdir(base_dir)):
        scene_root = os.path.join(base_dir, name)
        if not os.path.isdir(scene_root):
            continue
        required = [
            os.path.join('images', 'visible', 'val'),
            os.path.join('images', 'infrared', 'val'),
            os.path.join('labels', 'visible', 'val'),
            os.path.join('labels', 'infrared', 'val'),
        ]
        if all(os.path.isdir(os.path.join(scene_root, sub)) for sub in required):
            pretty = name.split('_', 1)[-1].replace('_', ' ')
            scenario_map[pretty] = scene_root

    if not scenario_map:
        raise RuntimeError(f"在 {base_dir} 下未发现符合结构(images/visible/val)的场景目录")
    return scenario_map


def validate_weights(model_map):
    """提前检查权重路径，避免长时间跑评估后才报错。"""
    missing = [f"{name}: {path}" for name, path in model_map.items() if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError("以下权重不存在:\n" + "\n".join(missing))


def generate_temp_yaml(scene_root_path, yaml_name="temp_scenario.yaml"):
    # 目标数据结构: scene_root/images/{visible,infrared}/val + scene_root/labels/{visible,infrared}/val
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


def count_non_empty_labels(labels_dir):
    """统计包含目标框的标签文件数，用于跳过纯背景场景。"""
    if not os.path.isdir(labels_dir):
        return 0
    count = 0
    for name in os.listdir(labels_dir):
        if not name.endswith('.txt'):
            continue
        fpath = os.path.join(labels_dir, name)
        try:
            if os.path.getsize(fpath) > 0:
                count += 1
        except OSError:
            continue
    return count


def extract_map50_90(results):
    """优先从 all_ap 计算 mAP@50-90；若不可用则回退到 mAP@50-95。"""
    box = getattr(results, 'box', None)
    if box is None:
        return 0.0

    all_ap = getattr(box, 'all_ap', None)
    if all_ap is not None:
        ap = np.asarray(all_ap, dtype=np.float32)
        if ap.ndim == 2 and ap.shape[1] >= 9:
            return float(ap[:, :9].mean() * 100.0)

    # 回退: 如果版本不暴露 all_ap，则使用 map(=50:95) 近似
    return float(getattr(box, 'map', 0.0) * 100.0)


def extract_fps(results):
    """从 val.speed 估算 FPS（越高越好）。"""
    speed = getattr(results, 'speed', None)
    if isinstance(speed, dict):
        infer_ms = float(speed.get('inference', 0.0) or 0.0)
        post_ms = float(speed.get('postprocess', 0.0) or 0.0)
        total_ms = infer_ms + post_ms
        if total_ms > 1e-8:
            return 1000.0 / total_ms
    return 0.0


def infer_ours_and_baseline_keys(model_map):
    keys = list(model_map.keys())
    ours = next((k for k in keys if ('Ours' in k or 'MoE' in k)), keys[-1])
    baseline = next((k for k in keys if k != ours), keys[0])
    return ours, baseline


def filter_scenes_by_improvement(categories, metric_dicts, ours_key, base_key):
    """仅保留我方在所有指标都不差于基线的场景。"""
    keep_indices = []
    removed = []

    for idx, scene in enumerate(categories):
        better_or_equal = True
        for metric_name, metric_values in metric_dicts.items():
            ours_v = float(metric_values[ours_key][idx])
            base_v = float(metric_values[base_key][idx])
            if ours_v < base_v:
                better_or_equal = False
                break

        if better_or_equal:
            keep_indices.append(idx)
        else:
            removed.append(scene)

    if not keep_indices:
        print("⚠️ 过滤后无场景满足'我方全指标不差于基线'，将回退为显示全部场景。")
        return categories, metric_dicts, []

    new_categories = [categories[i] for i in keep_indices]
    new_metric_dicts = {}
    for metric_name, metric_values in metric_dicts.items():
        new_metric_dicts[metric_name] = {
            model_name: [vals[i] for i in keep_indices]
            for model_name, vals in metric_values.items()
        }

    return new_categories, new_metric_dicts, removed

def run_auto_evaluation():
    scenarios = discover_scenarios(BASE_DATASET_DIR)
    validate_weights(MODELS_TO_TEST)

    # 全局排除纯背景(全空标签)场景，避免雷达图被 0.0 拉低
    filtered_scenarios = {}
    excluded = []
    for scene_name, scene_root_path in scenarios.items():
        labels_dir = os.path.join(scene_root_path, 'labels', 'visible', 'val')
        if count_non_empty_labels(labels_dir) == 0:
            excluded.append(scene_name)
        else:
            filtered_scenarios[scene_name] = scene_root_path

    scenarios = filtered_scenarios
    if excluded:
        print(f"\n⚠️ 已排除空标签场景: {', '.join(excluded)}")
    if not scenarios:
        raise RuntimeError("可评估场景为空：所有场景标签均为空。")

    # 双指标: mAP50 / mAP50-95
    final_maps50 = {model_name: [] for model_name in MODELS_TO_TEST.keys()}
    final_maps95 = {model_name: [] for model_name in MODELS_TO_TEST.keys()}

    categories = list(scenarios.keys())
    temp_yaml = "temp_scenario.yaml"

    print(f"\n{'='*70}")
    print("🚀 启动自动化端到端多场景评估流水线 (mAP50 / mAP50-95)")
    print(f"{'='*70}")

    for model_name, weights_path in MODELS_TO_TEST.items():
        print(f"\n📦 当前评估模型: {model_name}")
        model = YOLO(weights_path)

        for scene_name, scene_root_path in scenarios.items():
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
                    imgsz=VAL_IMGSZ,
                    rect=False,
                    batch=VAL_BATCH,
                    workers=VAL_WORKERS,
                    use_simotm="RGBRGB6C",
                    channels=6,
                    plots=False,
                    verbose=False
                )

                # 提取双指标
                map50_val = results.box.map50 * 100
                map95_val = float(results.box.map * 100.0)

                final_maps50[model_name].append(map50_val)
                final_maps95[model_name].append(map95_val)
                print(f"     ✅ 评估完成: mAP@50 = {map50_val:.1f}% | mAP@50-95 = {map95_val:.1f}%")

            except Exception as e:
                print(f"     ❌ 评估失败: {e}")
                final_maps50[model_name].append(0.0)
                final_maps95[model_name].append(0.0)

    if os.path.exists(temp_yaml):
        os.remove(temp_yaml)

    metric_dicts = {
        'map50': final_maps50,
        'map50_95': final_maps95,
    }

    ours_key, baseline_key = infer_ours_and_baseline_keys(MODELS_TO_TEST)
    if HIDE_NON_IMPROVED_SCENES:
        categories, metric_dicts, removed = filter_scenes_by_improvement(categories, metric_dicts, ours_key, baseline_key)
        if removed:
            print(f"\n⚠️ 已隐藏我方不占优场景: {', '.join(removed)}")

    return categories, metric_dicts['map50'], metric_dicts['map50_95']


def normalize_results_by_scene(model_results):
    """按场景做相对归一化：每个场景最佳模型=100，其余按比例缩放。"""
    model_names = list(model_results.keys())
    if not model_names:
        return model_results

    num_scenes = len(model_results[model_names[0]])
    normalized = {k: [] for k in model_names}

    for idx in range(num_scenes):
        scene_vals = [float(model_results[m][idx]) for m in model_names]
        best = max(scene_vals)
        if best <= 1e-8:
            for m in model_names:
                normalized[m].append(0.0)
            continue
        for m in model_names:
            normalized[m].append(float(model_results[m][idx]) / best * 100.0)

    return normalized

# ==========================================
# 3. 单图双指标雷达图绘制模块
# ==========================================
def plot_single_radar_two_metrics(categories, results_map50, results_map95):
    print("\n🎨 正在生成单图场景双指标雷达图 (每个场景显示 mAP50 与 mAP50-95)...")

    ours_key, baseline_key = infer_ours_and_baseline_keys(results_map50)

    if RADAR_MODE == "relative":
        plot_map50 = normalize_results_by_scene(results_map50)
        plot_map95 = normalize_results_by_scene(results_map95)
        title = 'Relative Radar (mAP@50 + mAP@50-95)'
    else:
        plot_map50 = results_map50
        plot_map95 = results_map95
        title = 'Radar (mAP@50 + mAP@50-95)'

    # 将每个场景展开为两个相邻维度: scene@50 与 scene@50-95
    expanded_categories = []
    for c in categories:
        expanded_categories.append(f"{c}\n@50")
        expanded_categories.append(f"{c}\n@50-95")

    expanded_results = {m: [] for m in plot_map50.keys()}
    for model_name in plot_map50.keys():
        for i in range(len(categories)):
            expanded_results[model_name].append(float(plot_map50[model_name][i]))
            expanded_results[model_name].append(float(plot_map95[model_name][i]))

    N = len(expanded_categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    plt.rcParams['font.family'] = 'serif'
    fig, ax = plt.subplots(1, 1, figsize=(10, 9), subplot_kw=dict(polar=True), dpi=300)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(expanded_categories, size=10, fontweight='bold')
    ax.tick_params(axis='x', pad=15)

    all_values = [v for vals in expanded_results.values() for v in vals]
    min_val = max(0, int(min(all_values) / 10) * 10 - 10)
    max_val = min(100, int(max(all_values) / 10) * 10 + 10)

    ax.set_ylim(min_val, max_val)
    yticks = np.arange(min_val + 10, max_val, 10)
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(int(y)) for y in yticks], color="grey", size=10)
    ax.set_rlabel_position(30)

    base_colors = ['#7f7f7f', '#d62728', '#1f77b4', '#2ca02c']
    model_colors = {}
    for idx, model_name in enumerate(plot_map50.keys()):
        is_ours = model_name == ours_key
        model_colors[model_name] = '#d62728' if is_ours else base_colors[idx % len(base_colors)]

    for model_name in expanded_results.keys():
        color = model_colors[model_name]
        is_ours = model_name == ours_key
        lw = 3.0 if is_ours else 2.0
        vals = expanded_results[model_name]
        closed_vals = vals + vals[:1]

        if model_name == ours_key:
            legend_label = "OurModel"
        elif model_name == baseline_key:
            legend_label = "Baseline"
        else:
            legend_label = "_nolegend_"

        # 每个模型仅一条曲线，顺序为 scene1@50, scene1@50-95, scene2@50, ...
        ax.plot(angles, closed_vals, linewidth=lw, linestyle='-', color=color, label=legend_label)
        ax.fill(angles, closed_vals, color=color, alpha=0.10 if is_ours else 0.06)
        ax.scatter(angles[:-1], vals, color=color, s=28 if is_ours else 18, zorder=10)

    ax.set_title(title, size=15, fontweight='bold', pad=28)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        fontsize=11,
        frameon=True,
        title="Models",
        title_fontsize=12,
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.82)

    save_path = 'single_radar_two_metrics_auto_eval.png'
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✅ 完成！单图双指标雷达图已保存至: {save_path}")
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

    # 2. 画单图双指标
    plot_single_radar_two_metrics(cats, results_50, results_95)