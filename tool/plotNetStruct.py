from ultralytics import YOLO
# 加载yaml配置文件
model = YOLO("/home/adrianyan/user/study/YOLOv11-RGBT/ultralytics/cfg/models/11-RGBT/yolo11-RGBT-midfusion.yaml")  # 替换为你的yaml文件路径
# 可视化网络结构，保存为pdf/图片
model.model.fuse()  # 融合层（可选，简化结构）
model.summary()     # 打印网络层详情（参数量/计算量/输出维度）
model.plot(filename="/home/adrianyan/user/study/YOLOv11-RGBT/result/net_struct/yolo_net_struct.png")  # 保存结构可视化图
