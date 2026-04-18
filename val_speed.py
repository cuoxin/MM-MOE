import warnings
warnings.filterwarnings('ignore')
import torch
import numpy as np
from ultralytics import YOLO

def measure_standard_speed(weights_path, imgsz=640, channels=6, batch_size=1, warmup_runs=100, test_runs=1000):
    """
    严谨的学术级模型测速模块
    :param weights_path: 模型权重路径
    :param imgsz: 输入图像尺寸 (默认 640)
    :param channels: 输入通道数 (双流拼接为 6)
    :param batch_size: 批大小 (评测端侧/无人机延迟通常用 batch=1)
    :param warmup_runs: 暖机运行次数 (唤醒 GPU 睡眠状态)
    :param test_runs: 实际测速运行次数 (求平均，消除抖动)
    """
    print(f"\n{'='*60}")
    print(f"🚀 启动标准测速模块 (Standard Speed Benchmark)")
    print(f"📦 权重路径: {weights_path}")
    print(f"⚙️  测试配置: Input=[{batch_size}, {channels}, {imgsz}, {imgsz}], Warmup={warmup_runs}, Test={test_runs}")
    print(f"{'='*60}\n")

    # 1. 初始化设备与模型
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print("⚠️ 警告: 当前正在使用 CPU 测速，这无法反映真实的硬件加速性能！")

    print("加载模型中...")
    yolo_model = YOLO(weights_path)
    # 提取纯 PyTorch 底层模型，并设置为 eval 模式
    pytorch_model = yolo_model.model.to(device).eval()

    # 2. 构造极其纯粹的模拟输入 (Dummy Input)
    # 剥离了硬盘读写、OpenCV 解码等所有外部干扰
    dummy_input = torch.randn(batch_size, channels, imgsz, imgsz, device=device, dtype=torch.float32)

    # 3. GPU 暖机 (Warm-up)
    # 刚启动时，GPU 处于低频节能状态，前几十次推理会异常缓慢。必须先把它“跑热”。
    print("🔥 正在执行 GPU 暖机 (Warm-up)...")
    with torch.no_grad():
        for i in range(warmup_runs):
            _ = pytorch_model(dummy_input)
            if (i + 1) % (warmup_runs // 2) == 0:
                print(f"   暖机进度: {i + 1}/{warmup_runs}")

    # 4. 正式测速 (使用严格的 CUDA Event)
    print("\n⏱️ 开始正式测速 (使用 CUDA 硬件级同步计时)...")
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = np.zeros((test_runs, 1))

    with torch.no_grad():
        for i in range(test_runs):
            # 等待上一步所有 CUDA 核心彻底完工
            torch.cuda.synchronize()

            # 打上开始时间戳
            starter.record()

            # 纯粹的网络前向传播 (不含 NMS 后处理)
            _ = pytorch_model(dummy_input)

            # 打上结束时间戳
            ender.record()

            # 强制等待本次推理彻底完成再算时间
            torch.cuda.synchronize()

            # 计算毫秒耗时
            curr_time = starter.elapsed_time(ender)
            timings[i] = curr_time

    # 5. 计算严谨的统计学指标
    # 掐头去尾：去掉最快和最慢的 5% 的极端系统抖动数据，求中间 90% 的平均值
    p5 = np.percentile(timings, 5)
    p95 = np.percentile(timings, 95)
    filtered_timings = timings[(timings >= p5) & (timings <= p95)]

    avg_latency = np.mean(filtered_timings)
    std_latency = np.std(filtered_timings)
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0

    print(f"\n{'='*60}")
    print(f"📊 测速最终报告 (Benchmark Report)")
    print(f"{'='*60}")
    print(f"🔸 单图平均推理延迟 (Avg Latency) : {avg_latency:.2f} ms  (± {std_latency:.2f} ms)")
    print(f"🔸 系统等效吞吐量 (FPS)           : {fps:.1f} Frames Per Second")
    print(f"🔸 [备注: 掐头去尾排除了 {test_runs - len(filtered_timings)} 次系统抖动异常值]")
    print(f"{'='*60}\n")

    return avg_latency, fps

if __name__ == '__main__':
    # 你的验证集评测代码可以保留在这里，作为精度的参考
    # ... 原本的 model.val(...) ...

    # ===============================================
    # 插入标准测速模块
    # ===============================================
    MODEL_WEIGHTS = r'/root/autodl-tmp/MM-MOE/runs/YOLOv12/myDualData_RGBRGB_midfusion_P3/weights/best.pt'

    # 执行测速，注意通道数设为 6 (对应你的 RGBRGB6C)
    measure_standard_speed(
        weights_path=MODEL_WEIGHTS,
        imgsz=640,
        channels=6,
        batch_size=1,     # 无人机端侧部署标准通常测 Batch=1 的延迟
        warmup_runs=100,  # 暖机 100 次
        test_runs=1000    # 正式跑 1000 次求平稳均值
    )