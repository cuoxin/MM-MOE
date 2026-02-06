from ultralytics import YOLO
from ultralytics.nn.modules.moe.stats import MoEStatsRecorder

# 1. 加载训练好的模型
model = YOLO(r"/root/autodl-tmp/MM-MOE/runs/myVisDrone/myVisDrone-yolo11n-MMMOE-test3/weights/best.pt")

# 2. 开启统计收集器
recorder = MoEStatsRecorder()
recorder.start_collection()

# 3. 运行验证 (跑一遍测试集)
# 这会自动调用 forward，触发埋点

metrics = model.val(data=r'/root/autodl-tmp/MM-MOE/ultralytics/cfg/datasets/myVisDrone.yaml',
            split='val',
            imgsz=640,
            batch=16,
            # use_simotm="RGBT",  # 4通道 RGB + IR
            # channels=4,

            use_simotm="RGBRGB6C", # 6 通道 RGB + IR(3通道)
            channels=6,

            # use_simotm="RGB",  # 3 通道 RGB 其余模式类似
            # channels=3,

            # pairs_rgb_ir=['visible','infrared'] , # default: ['visible','infrared'] , others: ['rgb', 'ir'],  ['images', 'images_ir'], ['images', 'image']
            # rect=False,
            # save_json=True, # if you need to cal coco metrice
            project='runs/val/myVisDrone',
            name='myVisDrone-yolo11n-MMMOE-test3',
            )

# 4. 保存统计结果
recorder.save_report(filename='0206_moe_stats_myVisDrone.csv')