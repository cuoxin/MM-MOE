'''
深层特征抗噪可视化,通过 Grad-CAM 展示 V20 异构 MoE 模型在强反光场景下的 P5 层双流分支的特征响应对比。
'''

import warnings
warnings.filterwarnings('ignore')
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


# ==========================================
# 1. 核心配置区域
# ==========================================
# 1.1 模型权重
WEIGHTS_PATH_V20 = r'/root/autodl-tmp/MM-MOE/runs/myDualData/myDualData-yolo11n-MMMOE-backboneV1_20-test-e300-topk1-0117-/weights/best.pt'
# 建议找一个 V19 (无直通专家) 的权重做对比，效果最震撼。
# 如果没有 V19，可以用 Baseline 权重：
WEIGHTS_PATH_BASELINE = r'/root/autodl-tmp/MM-MOE/runs/myDualData/myDualData-yolo11n-baseline-test-03132/weights/best.pt'

# 选取一张你想要可视化的极端场景测试图片 (例如强反光、夜间)
RGB_IMG_PATH = r'/root/autodl-tmp/datasets/sorted_scenes/3_SeaFog/visible/229_0001_183_base.jpg'
IR_IMG_PATH  = r'/root/autodl-tmp/datasets/sorted_scenes/3_SeaFog/infrared/229_0001_183_base.jpg'

# 1.3 层号严格对齐配置 🌟
# Baseline 模型的 P5 层双分支
LAYER_BASE_BRANCH_1 = 10  # Baseline 可见光 P5 (A2C2f)
LAYER_BASE_BRANCH_2 = 20  # Baseline 红外 P5 (A2C2f)

# V20 模型的 P5 层双分支
LAYER_V20_BRANCH_1 = 12   # V20 可见光 P5 MoE
LAYER_V20_BRANCH_2 = 24   # V20 红外 P5 MoE

# ==========================================
# 2. YOLO 梯度包装器
# ==========================================
class YOLOWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        for param in self.model.parameters():
            param.requires_grad_(True)

    def forward(self, x):
        x.requires_grad_(True)
        preds = self.model(x)
        if isinstance(preds, tuple) or isinstance(preds, list):
            preds = preds[0]
        class_scores = preds[:, 4:, :]
        max_scores, _ = torch.max(class_scores, dim=2)
        return max_scores

# ==========================================
# 3. 核心提取逻辑
# ==========================================
def extract_gradcam(model_path, input_tensor, target_layer_idx, is_moe=False):
    """提取核心 Grad-CAM 热力图 (返回黑白权重矩阵)"""
    yolo_model = YOLO(model_path)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    pytorch_model = yolo_model.model.to(device)

    wrapped_model = YOLOWrapper(pytorch_model).train()

    try:
        if is_moe:
            # V20 MoE 模型需要向下钻取
            target_layer = wrapped_model.model.model[target_layer_idx].experts
        else:
            # Baseline 模型直接取层
            target_layer = wrapped_model.model.model[target_layer_idx]
    except AttributeError:
        target_layer = wrapped_model.model.model[target_layer_idx]

    # 关注类别 0 (船/人)
    targets = [ClassifierOutputTarget(0)]
    cam = GradCAM(model=wrapped_model, target_layers=[target_layer])

    heatmap_grayscale = cam(input_tensor=input_tensor, targets=targets)[0, :]

    del cam, wrapped_model, yolo_model
    torch.cuda.empty_cache()

    return heatmap_grayscale

# ==========================================
# 4. 主执行与绘图排版
# ==========================================
if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    img1 = cv2.imread(RGB_IMG_PATH)
    img2 = cv2.imread(IR_IMG_PATH)
    if img1 is None or img2 is None:
        raise ValueError("图片读取失败！请检查路径。")

    img_rgb_arr = cv2.cvtColor(cv2.resize(img1, (640, 640)), cv2.COLOR_BGR2RGB)
    img_ir_arr  = cv2.cvtColor(cv2.resize(img2, (640, 640)), cv2.COLOR_BGR2RGB)

    rgb_float = img_rgb_arr.astype(np.float32) / 255.0
    ir_float  = img_ir_arr.astype(np.float32)  / 255.0

    t1 = torch.from_numpy(img_rgb_arr).permute(2, 0, 1).float() / 255.0
    t2 = torch.from_numpy(img_ir_arr).permute(2, 0, 1).float()  / 255.0
    input_tensor = torch.cat((t1, t2), dim=0).unsqueeze(0).to(device)

    print("🧠 1. 正在提取 Baseline 模型的双分支特征图...")
    cam_base_b1 = extract_gradcam(WEIGHTS_PATH_BASELINE, input_tensor, LAYER_BASE_BRANCH_1, is_moe=False)
    cam_base_b2 = extract_gradcam(WEIGHTS_PATH_BASELINE, input_tensor, LAYER_BASE_BRANCH_2, is_moe=False)

    print("🧠 2. 正在提取 V20 (Ours) 模型的双分支特征图...")
    cam_v20_b1 = extract_gradcam(WEIGHTS_PATH_V20, input_tensor, LAYER_V20_BRANCH_1, is_moe=True)
    cam_v20_b2 = extract_gradcam(WEIGHTS_PATH_V20, input_tensor, LAYER_V20_BRANCH_2, is_moe=True)

    print("🎨 3. 正在生成融合覆盖图...")
    # 可见光分支的热力图，只叠在 RGB 原图上
    vis_base_rgb = show_cam_on_image(rgb_float, cam_base_b1, use_rgb=True)
    vis_v20_rgb  = show_cam_on_image(rgb_float, cam_v20_b1, use_rgb=True)

    # 红外分支的热力图，只叠在 IR 原图上
    vis_base_ir  = show_cam_on_image(ir_float, cam_base_b2, use_rgb=True)
    vis_v20_ir   = show_cam_on_image(ir_float, cam_v20_b2, use_rgb=True)

    # ================= 排版绘制 2x3 绝杀对比图 =================
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 第一行：可见光视角 (RGB)
    axes[0, 0].imshow(img_rgb_arr)
    axes[0, 0].set_title("Input (RGB Visible)", fontsize=14)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(vis_base_rgb)
    axes[0, 1].set_title(f"Baseline Branch 1 (Layer {LAYER_BASE_BRANCH_1})", fontsize=14)
    axes[0, 1].axis('off')

    axes[0, 2].imshow(vis_v20_rgb)
    axes[0, 2].set_title(f"V20 Branch 1 (Layer {LAYER_V20_BRANCH_1}) - Clean Bg", fontsize=14, fontweight='bold', color='darkred')
    axes[0, 2].axis('off')

    # 第二行：红外视角 (IR)
    axes[1, 0].imshow(img_ir_arr)
    axes[1, 0].set_title("Input (Infrared)", fontsize=14)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(vis_base_ir)
    axes[1, 1].set_title(f"Baseline Branch 2 (Layer {LAYER_BASE_BRANCH_2})", fontsize=14)
    axes[1, 1].axis('off')

    axes[1, 2].imshow(vis_v20_ir)
    axes[1, 2].set_title(f"V20 Branch 2 (Layer {LAYER_V20_BRANCH_2}) - Target Focus", fontsize=14, fontweight='bold', color='darkred')
    axes[1, 2].axis('off')

    plt.tight_layout()
    save_path = "gradcam_dual_branch_compare.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 巅峰对比大图已保存至: {save_path}")
    plt.show()