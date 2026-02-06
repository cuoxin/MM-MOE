# ultralytics/nn/modules/moe/stats.py

import torch
import numpy as np
import csv
import os

class MoEStatsRecorder:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MoEStatsRecorder, cls).__new__(cls)
            cls._instance.reset()
        return cls._instance

    def reset(self):
        # 记录所有样本的专家选择索引
        self.expert_counts = {}  # {layer_id: [count_0, count_1, ...]}
        # 记录所有样本的路由权重
        self.router_weights = {} # {layer_id: [mean_w0, mean_w1, ...]}
        self.total_samples = 0
        self.collecting = False # 开关，默认关闭以免影响训练速度

    def start_collection(self):
        self.reset()
        self.collecting = True
        print("[MoE Stats] 开始收集路由数据...")

    def update(self, layer_id, indices, weights):
        """
        layer_id: 层的唯一标识 (e.g., 'P4_MoE')
        indices: [B, TopK] 专家索引
        weights: [B, TopK, C] 专家权重
        """
        if not self.collecting:
            return

        # 转为 CPU numpy
        idxs = indices.detach().cpu().numpy().flatten()

        # 初始化该层的统计
        if layer_id not in self.expert_counts:
            self.expert_counts[layer_id] = {}

        # 统计专家被选次数
        for i in idxs:
            self.expert_counts[layer_id][i] = self.expert_counts[layer_id].get(i, 0) + 1

    def save_report(self, save_dir='runs/test/stats', filename='moe_stats.csv'):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 保存为 CSV
        filepath = os.path.join(save_dir, filename)
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Layer', 'Expert_ID', 'Count', 'Percentage'])

            for layer, counts in self.expert_counts.items():
                total = sum(counts.values())
                for eid in sorted(counts.keys()):
                    writer.writerow([layer, eid, counts[eid], f"{counts[eid]/total*100:.2f}%"])

        print(f"[MoE Stats] 统计报告已保存至: {filepath}")