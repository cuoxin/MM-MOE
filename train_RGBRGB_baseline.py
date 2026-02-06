import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/root/autodl-tmp/MM-MOE/ultralytics/cfg/models/11-RGBT/yolo11-RGBRGB6C-midfusion.yaml')  # 只是将yaml里面的 ch设置成 6 ,红外部分改为 SilenceChannel, [ 3,6 ] 即可
    # model.load(r'yolo11n-RGBRGB6C-midfussion.pt') # loading pretrain weights 网盘下载
    model.train(data=R'/root/autodl-tmp/MM-MOE/ultralytics/cfg/datasets/myVisDrone.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=32,
                close_mosaic=0,
                workers=8,
                device='0',
                optimizer='SGD',  # using SGD
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                use_simotm="RGBRGB6C",
                channels=6,  #
                project='runs/myVisDrone',
                name='myVisDrone-yolo11n-baseline-test',
                # val=True,
                )