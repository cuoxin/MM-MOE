from ultralytics import YOLO

# 加载你训练好的模型
model = YOLO("/home/adrianyan/user/study/MM-MOE/run/result/myDrone-yolo11n-baseline-e300-0206.pt")

# 在你的硬件上进行基准测试 (imgsz应与训练一致)
model.benchmark(data='/home/adrianyan/user/study/MM-MOE/ultralytics/cfg/datasets/myVisDroneLocal.yaml',
                imgsz=640,
                device=0,
                split='test')