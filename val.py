import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(R'/home/adrianyan/user/study/MMOE_result/myDrone-yolo11n-MMMOE-backboneV1_10-e300-0224.pt')
    model.val(data=r'/home/adrianyan/user/study/MM-MOE/ultralytics/cfg/datasets/myVisDroneLocal.yaml',
              split='test',
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
              project='runs/val/myDrone',
              name='myDrone-yolo11n-MMMOE-backboneV1_10-e300-0225',
              )