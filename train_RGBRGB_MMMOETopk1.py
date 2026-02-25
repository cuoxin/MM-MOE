import torch
# 启用梯度异常检测，会打印详细的错误溯源
torch.autograd.set_detect_anomaly(True)

# import warnings
# warnings.filterwarnings('ignore')
from ultralytics import YOLO
from ultralytics.utils import LOGGER

def on_train_epoch_end(trainer):
    """
    在每个 Epoch 结束时调用。
    1. 打印专家分布到控制台
    2. 将专家分布保存到训练目录下的 moe_stats.txt
    3. 清零计数器
    """
    # -------------------------------------------------------
    # 1. 安全获取 Rank (防止多卡打印多次)
    # -------------------------------------------------------
    current_rank = getattr(trainer.args, 'rank', -1)
    if current_rank not in [-1, 0]:
        return

    # -------------------------------------------------------
    # 2. 准备日志文件路径 (自动跟随 project/name)
    # -------------------------------------------------------
    # trainer.save_dir 是 pathlib.Path 对象，指向 runs/project/name
    save_dir = trainer.save_dir
    log_file = save_dir / "moe_states.txt"

    # 准备要记录的文本内容
    header_msg = f"\n{'='*20} MoE Expert States (Epoch {trainer.epoch + 1}) {'='*20}\n"
    content_msgs = []

    found_router = False

    # 获取模型 (兼容 DDP)
    model = trainer.model
    if hasattr(model, 'module'):
        model = model.module

    # -------------------------------------------------------
    # 3. 遍历统计
    # -------------------------------------------------------
    for name, module in model.named_modules():
        if hasattr(module, 'selection_states') and hasattr(module, 'Layer_id'):
            found_router = True

            stats = module.selection_states
            total_calls = stats.sum().item()
            step_count = module.states_step_count.item()

            if total_calls > 0 and step_count > 0:
                # 转成百分比
                percentages = (stats / total_calls * 100).cpu().tolist()
                states_str = " | ".join([f"Exp{i}: {p:5.1f}%" for i, p in enumerate(percentages)])

                avg_scores = (module.expert_scores_sum / step_count).cpu().tolist()
                scores_str = " | ".join([f"{s:5.2f}" for s in avg_scores])

                msg_line1 = f"Layer {module.Layer_id} [Select%]: {states_str}"
                msg_line2 = f"       >>> [Avg Score]: {scores_str}"
                content_msgs.append(msg_line1)
                content_msgs.append(msg_line2)
                content_msgs.append("-"*60)
            else:
                content_msgs.append(f"Layer {module.Layer_id}: No data (total_calls=0)")

            # 🔥 必须清零
            module.selection_states.zero_()
            module.expert_scores_sum.zero_()
            module.states_step_count.zero_()

    footer_msg = "="*60 + "\n"

    # -------------------------------------------------------
    # 4. 执行打印和保存
    # -------------------------------------------------------
    if found_router:
        # A. 控制台打印
        LOGGER.info(header_msg.strip())
        for msg in content_msgs:
            LOGGER.info(msg)
        LOGGER.info(footer_msg.strip())

        # B. 写入文件 (追加模式)
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(header_msg)
                for msg in content_msgs:
                    f.write(msg + "\n")
                f.write(footer_msg + "\n")
        except Exception as e:
            LOGGER.warning(f"Failed to write MoE stats to file: {e}")
    else:
        LOGGER.info("No MoE Routers found to monitor.")

if __name__ == '__main__':
    model = YOLO('/root/autodl-tmp/MM-MOE/runs/myVisDrone/myVisDrone-yolo11n-MMMOE-backboneV1_11-test-e3-topk1-3/weights/last.pt')  # 只是将yaml里面的 ch设置成 6 ,红外部分改为 SilenceChannel, [ 3,6 ] 即可

    model.add_callback('on_train_epoch_end', on_train_epoch_end)

    model.train(data=R'/root/autodl-tmp/MM-MOE/ultralytics/cfg/datasets/myVisDrone.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=64,
                close_mosaic=10,
                workers=8,
                device='0',
                optimizer='SGD',  # using SGD
                resume=True,
                # amp=False, # close amp
                # fraction=0.2,
                use_simotm="RGBRGB6C",
                channels=6,  #
                project='runs/myVisDrone',
                name='myVisDrone-yolo11n-MMMOE-backboneV1_11-test-e3-topk1-',
                pretrained=False,
                amp=False
                )