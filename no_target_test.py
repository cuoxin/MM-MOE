import warnings
warnings.filterwarnings('ignore')
import torch
import time
import numpy as np
import cv2
from ultralytics import YOLO

def load_6ch_image(img_path_1, img_path_2, device, imgsz=(640, 640)):
    """
    加载两张对应的图片，将它们拼接成 1x6x640x640 的 Tensor
    """
    # 1. 读取图片 (cv2 默认读取为 BGR)
    img1 = cv2.imread(img_path_1)
    img2 = cv2.imread(img_path_2)

    if img1 is None or img2 is None:
        raise ValueError(f"图片读取失败，请检查路径: \n1: {img_path_1}\n2: {img_path_2}")

    # 2. 转换颜色空间 BGR -> RGB，并统一缩放尺寸
    img1 = cv2.cvtColor(cv2.resize(img1, imgsz), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.resize(img2, imgsz), cv2.COLOR_BGR2RGB)

    # 3. 将 NumPy 数组转为 Tensor，并归一化到 0-1 之间
    # 从 (H, W, C) 变成 (C, H, W)
    t1 = torch.from_numpy(img1).permute(2, 0, 1).float() / 255.0
    t2 = torch.from_numpy(img2).permute(2, 0, 1).float() / 255.0

    # 4. 在通道维度拼接 (3通道 + 3通道 = 6通道)
    tensor_6ch = torch.cat((t1, t2), dim=0)

    # 5. 增加 Batch 维度并放到对应设备上 (1, 6, H, W)
    tensor_6ch = tensor_6ch.unsqueeze(0).to(device)

    return tensor_6ch


def test_dynamic_speed():
    # 1. 加载你的 6 通道模型
    weights_path = r'/root/autodl-tmp/MM-MOE/runs/myDualData/myDualData-yolo11n-MMMOE-backboneV1_17-test-e300-topk1-0116-/weights/best.pt'
    model = YOLO(weights_path)

    # 确保模型在 GPU 上并设置为评估模式
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.model.to(device)
    model.model.eval()

    # 2. 构造真实图片数据
    # 🔴 请替换为你自己的真实图片路径
    print("正在加载图片数据...")

    # 场景一：纯背景图片 (无目标)
    bg_img_rgb = '/root/autodl-tmp/datasets/myDualData/images/visible/val/058_0001_584_neg.jpg'
    bg_img_ir = '/root/autodl-tmp/datasets/myDualData/images/infrared/val/058_0001_584_neg.jpg'
    bg_tensor = load_6ch_image(bg_img_rgb, bg_img_ir, device)

    # 场景二：复杂目标图片 (有目标，纹理复杂)
    target_img_rgb = '/root/autodl-tmp/datasets/myDualData/images/visible/val/924_0004_19_base.jpg'
    target_img_ir = '/root/autodl-tmp/datasets/myDualData/images/infrared/val/924_0004_19_base.jpg'
    complex_tensor = load_6ch_image(target_img_rgb, target_img_ir, device)

    # 3. 预热 GPU (防止第一次测速不准)
    print("正在预热模型...")
    with torch.no_grad():
        for _ in range(20):
            _ = model.model(bg_tensor)

    # 4. 测速函数
    def measure_latency(input_tensor, num_runs=100):
        latencies = []
        with torch.no_grad():
            for _ in range(num_runs):
                torch.cuda.synchronize()
                t0 = time.time()
                _ = model.model(input_tensor)
                torch.cuda.synchronize()
                t1 = time.time()
                latencies.append((t1 - t0) * 1000) # 转毫秒
        return np.mean(latencies[10:]) # 去掉前10次的波动

    # 5. 运行对比
    print("开始测试纯背景真实图片耗时...")
    bg_latency = measure_latency(bg_tensor)

    print("开始测试复杂目标真实图片耗时...")
    target_latency = measure_latency(complex_tensor)

    print("\n" + "="*40)
    print("🏎️ MoE 动态推理速度报告 (真实数据测试)")
    print("="*40)
    print(f"纯背景场景延迟 : {bg_latency:.2f} ms")
    print(f"复杂目标场景延迟 : {target_latency:.2f} ms")
    print(f"按需节省时间     : {target_latency - bg_latency:.2f} ms")

    if target_latency > bg_latency:
        print(f"最高可提速比例   : {((target_latency - bg_latency) / target_latency) * 100:.1f}%")
    else:
        print("提示：在此测试下，背景图片耗时未显著低于目标图片，说明 Router 的路径计算量近似。")
    print("="*40)

if __name__ == '__main__':
    test_dynamic_speed()