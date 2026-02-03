import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/home/adrianyan/user/study/YOLOv11-RGBT/ultralytics/cfg/models/11MMMOE/yolo11-RGBT-moe.yaml')  # 只是将yaml里面的 ch设置成 6 ,红外部分改为 SilenceChannel, [ 3,6 ] 即可
    # model.load(r'yolo11n-RGBRGB6C-midfussion.pt') # loading pretrain weights 网盘下载
    model.train(data=R'/home/adrianyan/user/study/YOLOv11-RGBT/ultralytics/cfg/datasets/FLIR_aligned.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=16,
                close_mosaic=0,
                workers=2,
                device='0',
                optimizer='SGD',  # using SGD
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                use_simotm="RGBRGB6C",
                channels=6,  #
                project='runs/FLIR_aligned_test',
                name='FLIR_aligned3C-yolo11n-MMMOE-test',
                # val=True,
                )