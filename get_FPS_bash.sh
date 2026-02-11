# 请根据实际情况修改 weights 路径和 channels
# --batch 1 代表测单张图片的延迟 (Latency)，最适合实时检测场景
# --testtime 500 代表跑 500 次取平均
python get_FPS.py \
  --weights root/autodl-tmp/MM-MOE/runs/myVisDrone/myVisDrone-yolo11n-MMMOE-topk1-test-e3-/weights/best.pt \
  --imgs 640 640 \
  --batch 1 \
  --channels 6 \
  --device 0 \
  --testtime 500