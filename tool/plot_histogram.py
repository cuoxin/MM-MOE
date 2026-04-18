"""
输出不同场景下 MoE 专家的决策分布柱状图。
- 场景来源与 plot_radar.py 一致：自动扫描 BASE_DATASET_DIR 下的 images/visible/val 结构。
- 自动排除全空标签场景（纯背景）。
- 统计 V8_1 的两个真实 MoE 层（P4=16, P5=19）的专家利用率与主导专家占比。
"""

import sys
import os

# 获取当前脚本所在目录的上一级目录 (即你的项目根目录)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 将根目录临时加入系统环境变量
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import warnings
warnings.filterwarnings('ignore')
import torch
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ==========================================
# 1. 核心配置区域 (请确认路径与你的服务器一致)
# ==========================================
WEIGHTS_PATH = r'/root/autodl-tmp/MM-MOE/runs/test_640/MMMOEV1_0_640/weights/best.pt'

BASE_DATASET_DIR = r'/root/autodl-tmp/datasets/test_sorted_scenes_640'
INPUT_IMGSZ = (640, 640)

# V6_0 实际 MoE 层索引（来自模型结构）
LAYER_P4 = 16
LAYER_P5 = 19


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
        ]
        if all(os.path.isdir(os.path.join(scene_root, sub)) for sub in required):
            pretty = name.split('_', 1)[-1].replace('_', ' ')
            scenario_map[pretty] = {
                'rgb': os.path.join(scene_root, 'images', 'visible', 'val'),
                'ir': os.path.join(scene_root, 'images', 'infrared', 'val'),
                'labels': os.path.join(scene_root, 'labels', 'visible', 'val'),
            }

    if not scenario_map:
        raise RuntimeError(f"在 {base_dir} 下未发现符合结构(images/visible/val)的场景目录")
    return scenario_map


def count_non_empty_labels(labels_dir):
    """统计包含目标框的标签文件数，用于排除纯背景场景。"""
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


def infer_num_experts(pytorch_model, layer_idx):
    """从模型层动态推断 routed expert 数，避免硬编码。"""
    layer = pytorch_model.model[layer_idx]
    experts_mod = layer.experts
    if hasattr(experts_mod, 'num_routed_experts'):
        return int(experts_mod.num_routed_experts)
    if hasattr(experts_mod, 'routed_experts'):
        return int(len(experts_mod.routed_experts))
    raise RuntimeError(f'无法从 layer {layer_idx} 推断专家数量，请检查 experts 模块实现。')

# ==========================================
# 2. 数据加载函数
# ==========================================
def letterbox_to_shape(img_bgr, target_hw=(640, 640), pad_value=114):
    """与训练/验证同口径: 保比例缩放 + padding 到固定输入尺寸。"""
    h, w = img_bgr.shape[:2]
    th, tw = target_hw

    r = min(tw / w, th / h)
    new_w = int(round(w * r))
    new_h = int(round(h * r))

    img = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    dw = tw - new_w
    dh = th - new_h
    top = dh // 2
    bottom = dh - top
    left = dw // 2
    right = dw - left

    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(pad_value, pad_value, pad_value))


def load_6ch_image(img_path_rgb, img_path_ir, device, imgsz=INPUT_IMGSZ):
    """加载 RGB/IR 并按训练同口径 letterbox 后拼接为 6 通道 Tensor。"""
    img1 = cv2.imread(img_path_rgb)
    img2 = cv2.imread(img_path_ir)
    if img1 is None or img2 is None:
        return None

    if isinstance(imgsz, int):
        target_hw = (imgsz, imgsz)
    else:
        target_hw = imgsz

    img1 = letterbox_to_shape(img1, target_hw=target_hw)
    img2 = letterbox_to_shape(img2, target_hw=target_hw)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    t1 = torch.from_numpy(img1).permute(2, 0, 1).float() / 255.0
    t2 = torch.from_numpy(img2).permute(2, 0, 1).float() / 255.0
    tensor_6ch = torch.cat((t1, t2), dim=0).unsqueeze(0).to(device)
    return tensor_6ch


def update_expert_counts(count_arr, dominant_arr, idx_tensor):
    """累计专家命中次数，并记录每张图的主导专家(idx[:,0])。"""
    idx_flat = idx_tensor.detach().view(-1).long().cpu().tolist()
    for idx in idx_flat:
        if 0 <= idx < len(count_arr):
            count_arr[idx] += 1

    idx_cpu = idx_tensor.detach().long().cpu()
    if idx_cpu.ndim == 2:
        dominant = idx_cpu[:, 0].view(-1).tolist()
    else:
        dominant = idx_cpu.view(-1).tolist()
    for idx in dominant:
        if 0 <= idx < len(dominant_arr):
            dominant_arr[idx] += 1

# ==========================================
# 3. 主统计与绘图逻辑 (Hook 大法)
# ==========================================
def generate_dual_histogram():
    print("🚀 正在初始化模型...")
    model = YOLO(WEIGHTS_PATH)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    pytorch_model = model.model.to(device).eval()

    scenarios = discover_scenarios(BASE_DATASET_DIR)
    empty_scenes = []
    for name, paths in scenarios.items():
        if count_non_empty_labels(paths['labels']) == 0:
            empty_scenes.append(name)

    if empty_scenes:
        print(f"ℹ️ 检测到空标签场景（将保留并统计背景路由行为）: {', '.join(empty_scenes)}")

    num_experts = infer_num_experts(pytorch_model, LAYER_P4)
    print(f"ℹ️ 动态检测到 routed experts = {num_experts}")

    # 初始化存储统计结果的字典
    p4_counts = {scene: np.zeros(num_experts) for scene in scenarios.keys()}
    p5_counts = {scene: np.zeros(num_experts) for scene in scenarios.keys()}
    p4_dominant = {scene: np.zeros(num_experts) for scene in scenarios.keys()}
    p5_dominant = {scene: np.zeros(num_experts) for scene in scenarios.keys()}

    # ---------------------------------------------------------
    # 🌟 核心：使用 PyTorch Hook 拦截专家路由决策 🌟
    # ---------------------------------------------------------
    current_idx_p4 = None
    current_idx_p5 = None

    def hook_p4(module, args, output):
        nonlocal current_idx_p4
        if len(args) >= 3 and torch.is_tensor(args[2]):
            current_idx_p4 = args[2]

    def hook_p5(module, args, output):
        nonlocal current_idx_p5
        if len(args) >= 3 and torch.is_tensor(args[2]):
            current_idx_p5 = args[2]

    print(f"🔗 正在挂载 Hook 拦截器到 MoE 层 (P4: {LAYER_P4}, P5: {LAYER_P5})...")
    try:
        handle1 = pytorch_model.model[LAYER_P4].experts.register_forward_hook(hook_p4)
        handle2 = pytorch_model.model[LAYER_P5].experts.register_forward_hook(hook_p5)
    except AttributeError:
        print("❌ 挂载 Hook 失败！请确保网络结构对应层级存在 `.experts` 属性。")
        return

    try:
        # ---------------------------------------------------------
        # 执行数据集遍历与推理
        # ---------------------------------------------------------
        print("📊 开始遍历场景数据集提取决策...")
        for scene_name, paths in scenarios.items():
            rgb_dir = paths['rgb']
            ir_dir = paths['ir']

            rgb_images = sorted(glob.glob(os.path.join(rgb_dir, "*.jpg")) + glob.glob(os.path.join(rgb_dir, "*.png")))
            if not rgb_images:
                print(f"⚠️ 警告: 在 {scene_name} 场景下未找到可见光图片，跳过。")
                continue

            print(f"  -> 分析 [{scene_name}]: 共 {len(rgb_images)} 张...")
            valid_count = 0

            for rgb_path in rgb_images:
                filename = os.path.basename(rgb_path)
                ir_path = os.path.join(ir_dir, filename)

                if not os.path.exists(ir_path):
                    continue

                tensor_6ch = load_6ch_image(rgb_path, ir_path, device)
                if tensor_6ch is None:
                    continue

                # 每次推理前重置状态
                current_idx_p4 = None
                current_idx_p5 = None

                # 前向传播 (自动触发 Hook，更新 current_idx)
                with torch.no_grad():
                    _ = pytorch_model(tensor_6ch)

                # 记录决策结果
                if current_idx_p4 is not None and current_idx_p5 is not None:
                    update_expert_counts(p4_counts[scene_name], p4_dominant[scene_name], current_idx_p4)
                    update_expert_counts(p5_counts[scene_name], p5_dominant[scene_name], current_idx_p5)
                    valid_count += 1

            print(f"     ✅ 成功统计 {valid_count} 张有效图像对。")
    finally:
        # 跑完数据后，卸载 Hook 释放内存
        handle1.remove()
        handle2.remove()

    # ---------------------------------------------------------
    # 计算百分比
    # ---------------------------------------------------------
    scenes = list(scenarios.keys())
    x = np.arange(len(scenes))
    width = 0.55 # 稍微加宽一点柱子，更好看

    p4_pct = []
    p4_dom_pct = []
    for scene in scenes:
        total = p4_counts[scene].sum()
        total_dom = p4_dominant[scene].sum()
        p4_pct.append((p4_counts[scene] / total * 100) if total > 0 else np.zeros(num_experts))
        p4_dom_pct.append((p4_dominant[scene] / total_dom * 100) if total_dom > 0 else np.zeros(num_experts))
    p4_pct = np.array(p4_pct)
    p4_dom_pct = np.array(p4_dom_pct)

    p5_pct = []
    p5_dom_pct = []
    for scene in scenes:
        total = p5_counts[scene].sum()
        total_dom = p5_dominant[scene].sum()
        p5_pct.append((p5_counts[scene] / total * 100) if total > 0 else np.zeros(num_experts))
        p5_dom_pct.append((p5_dominant[scene] / total_dom * 100) if total_dom > 0 else np.zeros(num_experts))
    p5_pct = np.array(p5_pct)
    p5_dom_pct = np.array(p5_dom_pct)

    # ---------------------------------------------------------
    # 🌟 终端打印详细统计报告 🌟
    # ---------------------------------------------------------
    print("\n" + "="*75)
    print("📝 文本版专家统计报告 (P4 / P5 MoE Layers)")
    print("="*75)
    for idx, scene in enumerate(scenes):
        total_routes = int(p4_counts[scene].sum())
        total_imgs = int(p4_dominant[scene].sum())
        print(f"【{scene}】有效测试图像对: {total_imgs} 张 | 总路由决策数: {total_routes}")
        if total_routes == 0:
            print("  (无数据)\n")
            continue

        p4_str_list = [f"Exp {i}: {int(p4_counts[scene][i]):>4}次 ({p4_pct[idx][i]:>5.1f}%)" for i in range(num_experts)]
        print(f"  [P4 Utilization] -> " + " | ".join(p4_str_list))
        p4_dom_str_list = [f"Exp {i}: {int(p4_dominant[scene][i]):>4}张 ({p4_dom_pct[idx][i]:>5.1f}%)" for i in range(num_experts)]
        print(f"  [P4 Dominant   ] -> " + " | ".join(p4_dom_str_list))

        p5_str_list = [f"Exp {i}: {int(p5_counts[scene][i]):>4}次 ({p5_pct[idx][i]:>5.1f}%)" for i in range(num_experts)]
        print(f"  [P5 Utilization] -> " + " | ".join(p5_str_list))
        p5_dom_str_list = [f"Exp {i}: {int(p5_dominant[scene][i]):>4}张 ({p5_dom_pct[idx][i]:>5.1f}%)" for i in range(num_experts)]
        print(f"  [P5 Dominant   ] -> " + " | ".join(p5_dom_str_list))
        print("-" * 75)
    print()

    # ---------------------------------------------------------
    # 🎨 绘制顶级学术风格堆叠柱状图
    # ---------------------------------------------------------
    print("🎨 正在生成双栏对比学术图表...")
    plt.rcParams['font.family'] = 'serif'

    # 颜色配置：Exp 0 (直通) 为冷色调，Exp 1~3 (计算密集) 为暖色渐变
    colors = ['#aec7e8', '#ffbb78', '#ff7f0e', '#d62728']
    labels = [f'Exp {i}' for i in range(num_experts)]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True, dpi=300)

    # -------- 左图: P4 MoE 主导专家占比 --------
    bottom1 = np.zeros(len(scenes))
    for i in range(num_experts):
        color = colors[i % len(colors)]
        axes[0].bar(x, p4_dom_pct[:, i], width, bottom=bottom1, label=labels[i], color=color, edgecolor='white', linewidth=1)
        bottom1 += p4_dom_pct[:, i]

    axes[0].set_title('P4 MoE Dominant Expert Assignment', fontsize=15, fontweight='bold', pad=15)
    axes[0].set_ylabel('Dominant Assignment (%)', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(scenes, fontsize=13)
    axes[0].grid(axis='y', linestyle='--', alpha=0.6)
    axes[0].tick_params(axis='y', labelsize=12)

    # -------- 右图: P5 MoE 主导专家占比 --------
    bottom2 = np.zeros(len(scenes))
    for i in range(num_experts):
        color = colors[i % len(colors)]
        axes[1].bar(x, p5_dom_pct[:, i], width, bottom=bottom2, label=labels[i], color=color, edgecolor='white', linewidth=1)
        bottom2 += p5_dom_pct[:, i]

    axes[1].set_title('P5 MoE Dominant Expert Assignment', fontsize=15, fontweight='bold', pad=15)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(scenes, fontsize=13)
    axes[1].grid(axis='y', linestyle='--', alpha=0.6)

    # -------- 统一图例与排版 --------
    # 将图例放在整个图表的正上方，横向排列
    fig.legend(labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=4, fontsize=12, frameon=True)

    plt.tight_layout()
    save_path = "global_routing_histogram_dominant.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 完美！学术级直方对比图已保存至: {save_path}")
    plt.show()

if __name__ == '__main__':
    generate_dual_histogram()