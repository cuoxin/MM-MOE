import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/root/autodl-tmp/MM-MOE/ultralytics/cfg/models/12/yolo12.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data=R'/root/autodl-tmp/MM-MOE/ultralytics/cfg/datasets/myDualDataI.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=64,
                close_mosaic=5,
                workers=8,
                device='0',
                optimizer='SGD',  # using SGD
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                use_simotm="RGB",
                channels=3,
                project='YOLOv12',
                name='myDualData_I_RGB',
                )