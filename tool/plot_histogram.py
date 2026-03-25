"""
输出不同层级的专家针对不同场景的决策分布柱状图，展示每个场景下各专家的利用率百分比。
- 场景包括：Normal, Night, Fog, StrongLight, Background
- 每个场景下分别统计可见光分支 (Branch 1) 和红外分支 (Branch 2) 的专家决策分布。
- 采用 PyTorch Forward Hook 技术，无损、无报错提取底层路由决策。
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
WEIGHTS_PATH = r'/root/autodl-tmp/MM-MOE/runs/myDualData/myDualData-yolo11n-MMMOE-backboneV2_22-test-e300-topk1-0119-13/weights/best.pt'

MMMOE_Lyars = {
    "P4": [9, 21],
    "P5": [12, 24]
}

LAYER_BRANCH_1 = MMMOE_Lyars["P4"][0]   # 分支 1 (可见光) 的层索引
LAYER_BRANCH_2 = MMMOE_Lyars["P4"][1]   # 分支 2 (红 外) 的层索引

# 场景测试集路径
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
# 2. 数据加载函数
# ==========================================
def load_6ch_image(img_path_rgb, img_path_ir, device, imgsz=(640, 640)):
    """加载 RGB 和 IR 图像并拼接为 6 通道 Tensor"""
    img1 = cv2.imread(img_path_rgb)
    img2 = cv2.imread(img_path_ir)
    if img1 is None or img2 is None:
        return None
    img1 = cv2.cvtColor(cv2.resize(img1, imgsz), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.resize(img2, imgsz), cv2.COLOR_BGR2RGB)
    t1 = torch.from_numpy(img1).permute(2, 0, 1).float() / 255.0
    t2 = torch.from_numpy(img2).permute(2, 0, 1).float() / 255.0
    tensor_6ch = torch.cat((t1, t2), dim=0).unsqueeze(0).to(device)
    return tensor_6ch

# ==========================================
# 3. 主统计与绘图逻辑 (Hook 大法)
# ==========================================
def generate_dual_histogram():
    print("🚀 正在初始化模型...")
    model = YOLO(WEIGHTS_PATH)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    pytorch_model = model.model.to(device).eval()

    # 初始化存储统计结果的字典
    b1_counts = {scene: np.zeros(4) for scene in SCENARIOS.keys()}
    b2_counts = {scene: np.zeros(4) for scene in SCENARIOS.keys()}

    # ---------------------------------------------------------
    # 🌟 核心：使用 PyTorch Hook 拦截专家路由决策 🌟
    # ---------------------------------------------------------
    current_idx_b1 = -1
    current_idx_b2 = -1

    def hook_branch_1(module, args, output):
        nonlocal current_idx_b1
        # args[2] 对应 DecoupledMoEContainer 的 forward(self, x, weights, indices) 中的 indices
        current_idx_b1 = int(args[2].item())

    def hook_branch_2(module, args, output):
        nonlocal current_idx_b2
        current_idx_b2 = int(args[2].item())

    print(f"🔗 正在挂载 Hook 拦截器到 P4 层 (Branch 1: {LAYER_BRANCH_1}, Branch 2: {LAYER_BRANCH_2})...")
    try:
        handle1 = pytorch_model.model[LAYER_BRANCH_1].experts.register_forward_hook(hook_branch_1)
        handle2 = pytorch_model.model[LAYER_BRANCH_2].experts.register_forward_hook(hook_branch_2)
    except AttributeError:
        print("❌ 挂载 Hook 失败！请确保网络结构对应层级存在 `.experts` 属性。")
        return

    # ---------------------------------------------------------
    # 执行数据集遍历与推理
    # ---------------------------------------------------------
    print("📊 开始遍历场景数据集提取决策...")
    for scene_name, paths in SCENARIOS.items():
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
            current_idx_b1 = -1
            current_idx_b2 = -1

            # 前向传播 (自动触发 Hook，更新 current_idx)
            with torch.no_grad():
                _ = pytorch_model(tensor_6ch)

            # 记录决策结果
            if current_idx_b1 != -1 and current_idx_b2 != -1:
                b1_counts[scene_name][current_idx_b1] += 1
                b2_counts[scene_name][current_idx_b2] += 1
                valid_count += 1
            else:
                print("⚠️ Hook 未能截获数据，网络数据流可能未经过该层级。")
                return

        print(f"     ✅ 成功统计 {valid_count} 张有效图像对。")

    # 跑完数据后，卸载 Hook 释放内存
    handle1.remove()
    handle2.remove()

    # ---------------------------------------------------------
    # 计算百分比
    # ---------------------------------------------------------
    scenes = list(SCENARIOS.keys())
    x = np.arange(len(scenes))
    width = 0.55 # 稍微加宽一点柱子，更好看

    b1_pct = []
    for scene in scenes:
        total = b1_counts[scene].sum()
        b1_pct.append((b1_counts[scene] / total * 100) if total > 0 else np.zeros(4))
    b1_pct = np.array(b1_pct)

    b2_pct = []
    for scene in scenes:
        total = b2_counts[scene].sum()
        b2_pct.append((b2_counts[scene] / total * 100) if total > 0 else np.zeros(4))
    b2_pct = np.array(b2_pct)

    # ---------------------------------------------------------
    # 🌟 终端打印详细统计报告 🌟
    # ---------------------------------------------------------
    print("\n" + "="*75)
    print("📝 文本版专家利用情况统计报告 (P4 Layer)")
    print("="*75)
    for idx, scene in enumerate(scenes):
        total_imgs = int(b1_counts[scene].sum())
        print(f"【{scene}】有效测试图像对: {total_imgs} 张")
        if total_imgs == 0:
            print("  (无数据)\n")
            continue

        b1_str_list = [f"Exp {i}: {int(b1_counts[scene][i]):>3}张 ({b1_pct[idx][i]:>5.1f}%)" for i in range(4)]
        print(f"  [可见光 Branch 1] -> " + " | ".join(b1_str_list))

        b2_str_list = [f"Exp {i}: {int(b2_counts[scene][i]):>3}张 ({b2_pct[idx][i]:>5.1f}%)" for i in range(4)]
        print(f"  [红  外 Branch 2] -> " + " | ".join(b2_str_list))
        print("-" * 75)
    print()

    # ---------------------------------------------------------
    # 🎨 绘制顶级学术风格堆叠柱状图
    # ---------------------------------------------------------
    print("🎨 正在生成双栏对比学术图表...")
    plt.rcParams['font.family'] = 'serif'

    # 颜色配置：Exp 0 (直通) 为冷色调，Exp 1~3 (计算密集) 为暖色渐变
    colors = ['#aec7e8', '#ffbb78', '#ff7f0e', '#d62728']
    labels = ['Exp 0 (Pass-Through)', 'Exp 1 (Active)', 'Exp 2 (Active)', 'Exp 3 (Active)']

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True, dpi=300)

    # -------- 左图: Branch 1 (可见光) --------
    bottom1 = np.zeros(len(scenes))
    for i in range(4):
        axes[0].bar(x, b1_pct[:, i], width, bottom=bottom1, label=labels[i], color=colors[i], edgecolor='white', linewidth=1)
        bottom1 += b1_pct[:, i]

    axes[0].set_title('Visible Branch (Branch 1) Expert Load', fontsize=15, fontweight='bold', pad=15)
    axes[0].set_ylabel('Allocation Percentage (%)', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(scenes, fontsize=13)
    axes[0].grid(axis='y', linestyle='--', alpha=0.6)
    axes[0].tick_params(axis='y', labelsize=12)

    # -------- 右图: Branch 2 (红外) --------
    bottom2 = np.zeros(len(scenes))
    for i in range(4):
        axes[1].bar(x, b2_pct[:, i], width, bottom=bottom2, label=labels[i], color=colors[i], edgecolor='white', linewidth=1)
        bottom2 += b2_pct[:, i]

    axes[1].set_title('Infrared Branch (Branch 2) Expert Load', fontsize=15, fontweight='bold', pad=15)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(scenes, fontsize=13)
    axes[1].grid(axis='y', linestyle='--', alpha=0.6)

    # -------- 统一图例与排版 --------
    # 将图例放在整个图表的正上方，横向排列
    fig.legend(labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=4, fontsize=12, frameon=True)

    plt.tight_layout()
    save_path = "global_routing_histogram.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 完美！学术级直方对比图已保存至: {save_path}")
    plt.show()

if __name__ == '__main__':
    generate_dual_histogram()