import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':

    '''
        source 为图像的最终目录,需要和train/val目录一致，且需要包含visible字段，visible同级目录下存在infrared目录，原理是将visible替换为infrared，加载双光谱数据

        "source" refers to the final directory for the images.
        The source needs to be in the same directory as the train/val directories, and it must contain the "visible" field.
        There is an "infrared" directory at the same level as the "visible" directory.
        The principle is to replace "visible" with "infrared" and load the dual-spectrum data.
    '''
    model = YOLO(r"/root/autodl-tmp/MM-MOE/runs/myDualData/myDualData-yolo11n-MMMOE-backboneV2_22-test-e300-topk1-0119-13/weights/best.pt") # select your model.pt path
    model.predict(source=r'/root/autodl-tmp/datasets/test_640/935_0001/visible',
                  imgsz=640,
                  project='runs/detect/935_0001',
                  name='exp',
                  show=False,
                  save_frames=True,
                  use_simotm="RGBRGB6C",
                  channels=6,
                  save=True,
                  conf=0.3,
                  save_txt=True,
                  # visualize=True # visualize model features maps
                )