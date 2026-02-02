from ultralytics import YOLO

model = YOLO("./ultralytics/cfg/models/master/v0_1/det/yolo-master-n.yaml")

results = model.train(
    data="./ultralytics/cfg/datasets/coco.yaml",
    epochs=600,
    batch=16,
    imgsz=640,
    device=0,
    scale=0.5,
    mosaic=1.0,
    mixup=0.0,
    copy_paste=0.1,
    workers=4,
    lr0=0.01,
    cache=False
)