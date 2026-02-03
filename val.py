import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(R'/home/adrianyan/user/study/YOLOv11-RGBT/result/old_weight/FLIR_aligned3C-yolo11n-RGBT-midfusion-e300-16-2.pt')
    model.val(data=r'/home/adrianyan/user/study/YOLOv11-RGBT/ultralytics/cfg/datasets/FLIR_aligned.yaml',
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
              project='runs/val/FLIR_aligned',
              name='FLIR_aligned3C-yolo11n-RGBT-midfusion-e300-16-2',
              )