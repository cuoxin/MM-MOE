import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_pt_experts(pt_path, moe_layer_names):
    # 加载权重
    ckpt = torch.load(pt_path, map_location='cpu')
    state_dict = ckpt['model'].state_dict() if 'model' in ckpt else ckpt

    for layer_name in moe_layer_names:
        print(f"\n分析层级: {layer_name}")
        expert_weights = []

        # 提取该层下所有专家的权重 (假设专家名为 routed_experts.0, .1 ...)
        # 请根据你 state_dict 里的实际 key 进行调整，通常是 .experts.routed_experts
        i = 0
        while True:
            # 针对你的 OptimizedSimpleExpert 结构，提取第一层 1x1 卷积权重
            key = f"{layer_name}.experts.routed_experts.{i}.conv.0.weight"
            if key in state_dict:
                w = state_dict[key].view(-1)
                expert_weights.append(w)
                i += 1
            else:
                break

        if not expert_weights:
            print(f"找不到权重，请检查 key: {layer_name}")
            continue

        num_experts = len(expert_weights)
        sim_matrix = torch.zeros((num_experts, num_experts))

        for i in range(num_experts):
            for j in range(num_experts):
                sim_matrix[i, j] = F.cosine_similarity(
                    expert_weights[i].unsqueeze(0),
                    expert_weights[j].unsqueeze(0)
                )

        # 打印矩阵
        print(sim_matrix)

        # 可视化
        plt.figure(figsize=(6, 5))
        sns.heatmap(sim_matrix.numpy(), annot=True, cmap='YlGnBu', vmin=0, vmax=1)
        plt.title(f"Expert Similarity - {layer_name}")
        plt.show()

# 使用方法：
analyze_pt_experts('/root/autodl-tmp/MM-MOE/runs/myDualData/myDualData-yolo11n-MMMOE-backboneV1_15-test-e300-topk1-0114-/weights/best.pt', ['model.9', 'model.12', 'model.21', 'model.24'])