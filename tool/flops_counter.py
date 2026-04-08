import warnings
warnings.filterwarnings('ignore')
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import torch
from ultralytics import YOLO
from thop import profile, clever_format
import warnings

# 忽略一些底层的追踪警告
warnings.filterwarnings('ignore')

def calculate_moe_params_and_flops(weight_path, imgsz=640, channels=6):
    """
    计算 MoE 模型的真实参数量和动态计算量。

    参数:
        weight_path: 你的 .pt 权重文件路径
        imgsz: 推理时的图像尺寸 (默认 640)
        channels: 输入通道数 (双模态 RGB+T 是 6 通道)
    """
    print(f"\n{'='*50}")
    print(f"🚀 正在加载权重文件: {weight_path}")

    # 1. 加载模型 (强制使用 CPU，避免显存干扰)
    device = torch.device('cpu')
    try:
        yolo = YOLO(weight_path)
        model = yolo.model.to(device)
        model.eval() # 必须切换到推理模式
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 2. 构造标准伪输入 (Dummy Input)
    # Batch Size 设置为 1，完全模拟单张图片推理的真实场景
    dummy_input = torch.randn(1, channels, imgsz, imgsz).to(device)

    print(f"📦 构建伪输入: Shape [1, {channels}, {imgsz}, {imgsz}]")
    print(f"⏳ 正在通过 thop 追踪动态计算图，请稍候...")

    # 3. 使用 thop 测算
    # 注意：因为你的 MoE 前向传播写了 `if not batch_mask.any(): continue`
    # 所以 thop 在追踪时，只会计算被激活的那个专家的 FLOPs！完美契合动态路由。
    macs, params = profile(model, inputs=(dummy_input, ), verbose=False)

    # 4. 格式化输出
    # MACs (Multiply-Accumulate Operations) 通常约等于 FLOPs 的一半
    # 业界通常直接把 MACs 乘以 2 作为 FLOPs，Ultralytics 官方 GFLOPs 也是这样算的
    flops = macs * 2.0

    # 转换为更易读的单位 (M = Million, G = Giga)
    params_m = params / 1e6
    flops_g = flops / 1e9

    print(f"\n{'='*50}")
    print("📊 [解耦对比分析报告]")
    print(f"   ➔ 总参数量 (Parameters) : {params_m:.3f} M")
    print(f"   ➔ 动态计算量 (GFLOPs)   : {flops_g:.3f} G")
    print(f"{'='*50}\n")

    return params_m, flops_g

if __name__ == "__main__":
    # 👇 在这里修改为你的权重路径

    # 1. 测算你的 Baseline 模型
    calculate_moe_params_and_flops("/root/autodl-tmp/MM-MOE/runs/myDualData/myDualData-yolo11n-baseline-test-03132/weights/best.pt", channels=6)

    # 2. 测算你的 MoE 最优模型 (YOLO1: 4专家, Top-1)
    # calculate_moe_params_and_flops("/root/autodl-tmp/MM-MOE/runs/myDualData/myDualData-yolo11n-MMMOE-backboneV2_22-test-e300-topk1-0119-/weights/best.pt", channels=6)