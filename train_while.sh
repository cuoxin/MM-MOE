#!/bin/bash

# ===================== 自定义训练参数列表 =====================
# 模型YAML路径列表（按需添加/删除）
MODEL_YAMLS=(
"/root/autodl-tmp/MM-MOE/ultralytics/cfg/models/11MMMOE/yolo11-RGBT-moe-backboneV7_0.yaml"
# "/root/autodl-tmp/MM-MOE/ultralytics/cfg/models/12-RGBT/yolo12-RGBT-midfusion-P3.yaml"
)

# 训练根目录(project)列表（与上面一一对应）
PROJECTS=(
"runs/myDualDataV4"
# "runs/myDualDataV4"
)

# 训练任务名称(name)列表（与上面一一对应）
NAMES=(
"myDualData-MMMOE-backbone-V7_0"
# "myDualData-baselineV2-P3"
)
# ==========================================================

# 检查三个列表长度是否一致
if [ ${#MODEL_YAMLS[@]} -ne ${#PROJECTS[@]} ] || [ ${#MODEL_YAMLS[@]} -ne ${#NAMES[@]} ]; then
    echo "错误：参数列表长度不一致！"
    exit 1
fi

# 循环执行训练任务
echo "========= 开始批量训练，总任务数：${#MODEL_YAMLS[@]} ========="
for ((i=0; i<${#MODEL_YAMLS[@]}; i++)); do
    echo -e "\n============================================="
    echo "执行第 $((i+1)) 个训练任务："
    echo "模型YAML：${MODEL_YAMLS[$i]}"
    echo "保存目录：${PROJECTS[$i]}"
    echo "任务名称：${NAMES[$i]}"
    echo "=============================================\n"

    # 调用Python训练脚本，传入参数
    python /root/autodl-tmp/MM-MOE/train_RGBRGB_while.py \
        --model-yaml "${MODEL_YAMLS[$i]}" \
        --project "${PROJECTS[$i]}" \
        --name "${NAMES[$i]}"

    # 检查任务是否执行失败
    if [ $? -ne 0 ]; then
        echo "第 $((i+1)) 个任务执行失败，继续下一个..."
    fi
done

echo -e "\n========= 所有批量训练任务执行完毕 ========="