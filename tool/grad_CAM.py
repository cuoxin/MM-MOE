'''
深层特征抗噪可视化。
通过 Grad-CAM 对比 Ours 与 Baseline(P3) 在 P4/P5 的响应差异，并输出热力图集中度指标。
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
WEIGHTS_PATH_V6 = r'/root/autodl-tmp/MM-MOE/runs/test_640/MMMOEV1_0_640/weights/best.pt'
# WEIGHTS_PATH_V6 = r'/home/adrianyan/user/study/MOE-weight/V3_0.pt'
# 建议找一个 V19 (无直通专家) 的权重做对比，效果最震撼。
# 如果没有 V19，可以用 Baseline 权重：
WEIGHTS_PATH_BASELINE = r'/root/autodl-tmp/MM-MOE/runs/all_data_V1_3/myDualData_RGBRGB_midfusion_P3/weights/best.pt'
# WEIGHTS_PATH_BASELINE = r'/home/adrianyan/user/study/MOE-weight/baseline.pt'

# 选取一张你想要可视化的极端场景测试图片 (例如强反光、夜间)
RGB_IMG_PATH = r'/root/autodl-tmp/datasets/test_sorted_scenes_640/3_SeaFog/images/visible/val/229_0001_182.jpg'
IR_IMG_PATH  = r'/root/autodl-tmp/datasets/test_sorted_scenes_640/3_SeaFog/images/infrared/val/229_0001_182.jpg'

# 1.3 层号严格对齐配置 🌟
# V6_0.yaml 中的 P4/P5 位置：
#   P4 基础特征: 15, P4 MoE: 16
#   P5 基础特征: 18, P5 MoE: 19
# Baseline 模型的 P4/P5 层
LAYER_BASE_P4 = 15  # Baseline P4 (A2C2f)
LAYER_BASE_P5 = 18  # Baseline P5 (A2C2f)

# V6_0 模型的 P4/P5 MoE 层
LAYER_V6_P4 = 16    # V6_0 P4 MoE
LAYER_V6_P5 = 19    # V6_0 P5 MoE

# 与训练脚本保持一致: imgsz=640
INPUT_IMGSZ = 640
TOP_RATIO_FOR_PEAK = 0.05

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

        # 兼容 Ultralytics 推理返回形式 (pred, aux) / [pred]
        if isinstance(preds, tuple) or isinstance(preds, list):
            preds = preds[0]

        if not torch.is_tensor(preds) or preds.ndim != 3:
            raise RuntimeError(f'Unexpected prediction format for Grad-CAM: type={type(preds)}, shape={getattr(preds, "shape", None)}')
        if preds.shape[1] <= 4:
            raise RuntimeError(f'Prediction channels invalid for class extraction: {preds.shape}')

        class_scores = preds[:, 4:, :]
        return torch.max(class_scores, dim=2).values

# ==========================================
# 3. 核心提取逻辑
# ==========================================
def get_model_stride(model_path):
    """读取模型最大 stride，用于自适应补边。"""
    yolo_model = YOLO(model_path)
    stride_attr = getattr(yolo_model.model, 'stride', torch.tensor([32]))
    stride = int(torch.as_tensor(stride_attr).max().item())
    del yolo_model
    return max(stride, 1)


def letterbox_min_pad(img_rgb, stride=32, target_hw=None, pad_value=114):
    """保持比例缩放并补边到 stride 对齐；返回补边后图像及几何参数。"""
    h, w = img_rgb.shape[:2]

    if target_hw is None:
        # 自适应到最近的 stride 倍数，避免 FPN/Concat 维度错位
        new_h = int(np.ceil(h / stride) * stride)
        new_w = int(np.ceil(w / stride) * stride)
    else:
        new_h, new_w = target_hw

    r = min(new_h / h, new_w / w)
    new_unpad_w = int(round(w * r))
    new_unpad_h = int(round(h * r))

    if (new_unpad_w, new_unpad_h) != (w, h):
        img_rgb = cv2.resize(img_rgb, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)

    dw = new_w - new_unpad_w
    dh = new_h - new_unpad_h

    if target_hw is None:
        # 上方已对齐到 stride 倍数，这里仅兜底避免负值
        dw = max(dw, 0)
        dh = max(dh, 0)

    top = dh // 2
    bottom = dh - top
    left = dw // 2
    right = dw - left

    img_rgb = cv2.copyMakeBorder(
        img_rgb, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(pad_value, pad_value, pad_value)
    )

    meta = {
        'orig_h': h,
        'orig_w': w,
        'unpadded_h': new_unpad_h,
        'unpadded_w': new_unpad_w,
        'top': top,
        'bottom': bottom,
        'left': left,
        'right': right,
    }
    return img_rgb, meta


def build_dual_input(rgb_bgr, ir_bgr, stride=32, imgsz=None):
    """构建 6 通道输入；与训练同口径: 固定640 letterbox。"""
    rgb_orig = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    ir_orig = cv2.cvtColor(ir_bgr, cv2.COLOR_BGR2RGB)

    if imgsz is None:
        imgsz = 640
    target_hw = (imgsz, imgsz)

    rgb, rgb_meta = letterbox_min_pad(rgb_orig, stride=stride, target_hw=target_hw)
    ir, ir_meta = letterbox_min_pad(ir_orig, stride=stride, target_hw=target_hw)

    if rgb.shape[:2] != ir.shape[:2]:
        raise ValueError(f"RGB/IR 预处理后尺寸不一致: RGB={rgb.shape[:2]}, IR={ir.shape[:2]}")

    t1 = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    t2 = torch.from_numpy(ir).permute(2, 0, 1).float() / 255.0
    input_tensor = torch.cat((t1, t2), dim=0).unsqueeze(0)

    return rgb_orig, ir_orig, input_tensor, rgb_meta, ir_meta


def remap_cam_to_original(cam_map, meta):
    """将 CAM 从补边空间映射回原图空间，避免边缘补边伪响应干扰。"""
    top = meta['top']
    left = meta['left']
    unpadded_h = meta['unpadded_h']
    unpadded_w = meta['unpadded_w']
    orig_h = meta['orig_h']
    orig_w = meta['orig_w']

    cam_crop = cam_map[top:top + unpadded_h, left:left + unpadded_w]
    if cam_crop.size == 0:
        raise ValueError('CAM 去补边后为空，请检查预处理参数。')

    cam_orig = cv2.resize(cam_crop, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    cam_orig = cam_orig.astype(np.float32)
    cam_orig -= cam_orig.min()
    cam_orig /= (cam_orig.max() + 1e-8)
    return cam_orig


def extract_gradcam(model_path, input_tensor, target_layer_idx, is_moe=False):
    """提取核心 Grad-CAM 热力图 (返回黑白权重矩阵)"""
    yolo_model = YOLO(model_path)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    pytorch_model = yolo_model.model.to(device)

    # CAM 对比使用推理态，避免 train/eval 输出语义不一致
    wrapped_model = YOLOWrapper(pytorch_model).eval()

    try:
        if is_moe:
            # V6_0 MoE 模型需要向下钻取
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


def calc_focus_metrics(cam_map, top_ratio=TOP_RATIO_FOR_PEAK):
    """输出热力图集中度指标：越低熵、越高峰值占比通常意味着注意力更集中。"""
    cam = cam_map.astype(np.float32)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    flat = cam.reshape(-1)
    prob = flat / (flat.sum() + 1e-8)
    entropy = float(-(prob * np.log(prob + 1e-12)).sum())

    k = max(1, int(len(flat) * top_ratio))
    peak_ratio = float(np.sort(flat)[-k:].sum() / (flat.sum() + 1e-8))

    return {
        'entropy': entropy,
        'peak_ratio': peak_ratio,
    }


def better_focus(ours_metrics, base_metrics):
    """集中度更好: 熵更低且峰值占比更高。"""
    return (ours_metrics['entropy'] < base_metrics['entropy']) and (ours_metrics['peak_ratio'] > base_metrics['peak_ratio'])

# ==========================================
# 4. 主执行与绘图排版
# ==========================================
if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    img1 = cv2.imread(RGB_IMG_PATH)
    img2 = cv2.imread(IR_IMG_PATH)
    if img1 is None or img2 is None:
        raise ValueError("图片读取失败！请检查路径。")

    model_stride = get_model_stride(WEIGHTS_PATH_V6)
    img_rgb_arr, img_ir_arr, input_tensor, rgb_meta, ir_meta = build_dual_input(
        img1, img2, stride=model_stride, imgsz=INPUT_IMGSZ
    )
    input_tensor = input_tensor.to(device)

    print(f"📐 使用输入尺寸: {img_rgb_arr.shape[1]}x{img_rgb_arr.shape[0]} (stride={model_stride}, imgsz={INPUT_IMGSZ})")

    print("🧠 1. 正在提取 Baseline 模型的 P4/P5 特征图...")
    cam_base_p4 = extract_gradcam(WEIGHTS_PATH_BASELINE, input_tensor, LAYER_BASE_P4, is_moe=False)
    cam_base_p5 = extract_gradcam(WEIGHTS_PATH_BASELINE, input_tensor, LAYER_BASE_P5, is_moe=False)

    print("🧠 2. 正在提取 V6_0 (Ours) 模型的 P4/P5 MoE 特征图...")
    cam_v6_p4 = extract_gradcam(WEIGHTS_PATH_V6, input_tensor, LAYER_V6_P4, is_moe=True)
    cam_v6_p5 = extract_gradcam(WEIGHTS_PATH_V6, input_tensor, LAYER_V6_P5, is_moe=True)

    cam_base_p4 = remap_cam_to_original(cam_base_p4, rgb_meta)
    cam_v6_p4 = remap_cam_to_original(cam_v6_p4, rgb_meta)
    cam_base_p5 = remap_cam_to_original(cam_base_p5, ir_meta)
    cam_v6_p5 = remap_cam_to_original(cam_v6_p5, ir_meta)

    m_base_p4 = calc_focus_metrics(cam_base_p4)
    m_ours_p4 = calc_focus_metrics(cam_v6_p4)
    m_base_p5 = calc_focus_metrics(cam_base_p5)
    m_ours_p5 = calc_focus_metrics(cam_v6_p5)

    print("\n📊 CAM 集中度指标（越低熵、越高峰值占比越集中）")
    print(f"P4  Baseline: entropy={m_base_p4['entropy']:.4f}, peak@{int(TOP_RATIO_FOR_PEAK*100)}%={m_base_p4['peak_ratio']:.4f}")
    print(f"P4  Ours    : entropy={m_ours_p4['entropy']:.4f}, peak@{int(TOP_RATIO_FOR_PEAK*100)}%={m_ours_p4['peak_ratio']:.4f}")
    print(f"P5  Baseline: entropy={m_base_p5['entropy']:.4f}, peak@{int(TOP_RATIO_FOR_PEAK*100)}%={m_base_p5['peak_ratio']:.4f}")
    print(f"P5  Ours    : entropy={m_ours_p5['entropy']:.4f}, peak@{int(TOP_RATIO_FOR_PEAK*100)}%={m_ours_p5['peak_ratio']:.4f}")

    p4_better = better_focus(m_ours_p4, m_base_p4)
    p5_better = better_focus(m_ours_p5, m_base_p5)
    print(f"结论: P4集中度更优={p4_better}, P5集中度更优={p5_better}\n")

    print("🎨 3. 正在生成融合覆盖图...")
    # P4 热力图叠在 RGB 原图上
    rgb_float = img_rgb_arr.astype(np.float32) / 255.0
    ir_float = img_ir_arr.astype(np.float32) / 255.0

    vis_base_p4_rgb = show_cam_on_image(rgb_float, cam_base_p4, use_rgb=True)
    vis_v6_p4_rgb   = show_cam_on_image(rgb_float, cam_v6_p4, use_rgb=True)

    # P5 热力图叠在 IR 原图上
    vis_base_p5_ir = show_cam_on_image(ir_float, cam_base_p5, use_rgb=True)
    vis_v6_p5_ir   = show_cam_on_image(ir_float, cam_v6_p5, use_rgb=True)

    # ================= 排版绘制 2x3 绝杀对比图 =================
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 第一行：P4 (RGB)
    axes[0, 0].imshow(img_rgb_arr)
    axes[0, 0].set_title("Input (RGB Visible)", fontsize=14)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(vis_base_p4_rgb)
    axes[0, 1].set_title(
        f"Baseline P4 (L{LAYER_BASE_P4})\\nH={m_base_p4['entropy']:.2f}, Peak={m_base_p4['peak_ratio']:.2f}",
        fontsize=13
    )
    axes[0, 1].axis('off')

    axes[0, 2].imshow(vis_v6_p4_rgb)
    axes[0, 2].set_title(
        f"Ours P4 MoE (L{LAYER_V6_P4})\\nH={m_ours_p4['entropy']:.2f}, Peak={m_ours_p4['peak_ratio']:.2f}",
        fontsize=13,
        fontweight='bold',
        color='darkred'
    )
    axes[0, 2].axis('off')

    # 第二行：P5 (IR)
    axes[1, 0].imshow(img_ir_arr)
    axes[1, 0].set_title("Input (Infrared)", fontsize=14)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(vis_base_p5_ir)
    axes[1, 1].set_title(
        f"Baseline P5 (L{LAYER_BASE_P5})\\nH={m_base_p5['entropy']:.2f}, Peak={m_base_p5['peak_ratio']:.2f}",
        fontsize=13
    )
    axes[1, 1].axis('off')

    axes[1, 2].imshow(vis_v6_p5_ir)
    axes[1, 2].set_title(
        f"Ours P5 MoE (L{LAYER_V6_P5})\\nH={m_ours_p5['entropy']:.2f}, Peak={m_ours_p5['peak_ratio']:.2f}",
        fontsize=13,
        fontweight='bold',
        color='darkred'
    )
    axes[1, 2].axis('off')

    plt.tight_layout()
    save_path = "gradcam_p4_p5_compare.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 巅峰对比大图已保存至: {save_path}")
    plt.show()