import torch

class MoEAuxCollector:
    _aux = []

    @classmethod
    def add(cls, aux):
        if aux is not None:
            # 直接存入 Tensor，保持梯度连通
            cls._aux.append(aux)

    @classmethod
    def pop_sum(cls, device=None, num_moe_layers=4):
        """
        弹出并求和。
        增加 num_moe_layers 参数，默认 4 个 MoE 层。
        """
        if not cls._aux:
            return None

        # 💥 核心修复：只取当前最新前向传播生成的最后 4 个 loss！
        # 完美扔掉 YOLO 初始化时留下的 "死节点(Dummy Loss)"
        valid_aux = cls._aux[-num_moe_layers:]

        if device is not None:
            processed_tensors = [a.to(device) for a in valid_aux]
        else:
            processed_tensors = valid_aux

        # 求和
        total_aux = torch.stack(processed_tensors).sum()

        # 清空列表，迎接下一个 Batch
        cls._aux = []

        return total_aux