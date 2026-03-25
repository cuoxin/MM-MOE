import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(R'/root/autodl-tmp/MM-MOE/runs/myDualData/myDualData-yolo11n-MMMOE-backboneV2_22-test-e300-topk1-0119-13/weights/best.pt')
    model.val(data=r'/root/autodl-tmp/MM-MOE/ultralytics/cfg/datasets/myDualData.yaml',
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
              project='runs/val/myDualData',
              name='myDualData-yolo11n-baseline-test-e300-0114-',
              )