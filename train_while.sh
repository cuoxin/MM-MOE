#!/bin/bash

# ===================== 自定义训练参数 =====================
# 模型目录：自动读取该目录下所有 .yaml
MODEL_DIR="/root/autodl-tmp/MM-MOE/ultralytics/cfg/models/12MMOE"

# 统一训练根目录：所有模型都保存到这个 project 下
PROJECT="runs/final_MMMOE"
# ==========================================================

if [ ! -d "$MODEL_DIR" ]; then
    echo "错误：模型目录不存在：$MODEL_DIR"
    exit 1
fi

# 自动收集模型，过滤掉以下划线开头的文件，例如 _MMOEV1_0.yaml
MODEL_YAMLS=()
while IFS= read -r -d '' file; do
    base_name="$(basename "$file")"
    if [[ "$base_name" == _* ]]; then
        continue
    fi
    MODEL_YAMLS+=("$file")
done < <(find "$MODEL_DIR" -maxdepth 1 -type f -name "*.yaml" -print0 | sort -z)

if [ ${#MODEL_YAMLS[@]} -eq 0 ]; then
    echo "错误：未找到可训练模型（已过滤下划线前缀文件）"
    exit 1
fi

# 循环执行训练任务
echo "========= 开始批量训练，总任务数：${#MODEL_YAMLS[@]} ========="
for ((i=0; i<${#MODEL_YAMLS[@]}; i++)); do
    model_yaml="${MODEL_YAMLS[$i]}"
    model_file="$(basename "$model_yaml")"
    # 自动提取 name：去掉 .yaml 后缀
    name="${model_file%.yaml}"

    echo -e "\n============================================="
    echo "执行第 $((i+1)) 个训练任务："
    echo "模型YAML：${model_yaml}"
    echo "保存目录：${PROJECT}"
    echo "任务名称：${name}"
    echo "=============================================\n"

    # 调用Python训练脚本，传入参数
    python /root/autodl-tmp/MM-MOE/train_RGBRGB_while.py \
        --model-yaml "${model_yaml}" \
        --project "${PROJECT}" \
        --name "${name}"

    # 检查任务是否执行失败
    if [ $? -ne 0 ]; then
        echo "第 $((i+1)) 个任务执行失败，继续下一个..."
    fi
done

echo -e "\n========= 所有批量训练任务执行完毕 ========="