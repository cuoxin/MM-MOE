import torch

# class MoEAuxCollector:
#     _aux = []

#     @classmethod
#     def add(cls, aux):
#         if aux is not None:
#             # 直接存入 Tensor，保持梯度连通
#             cls._aux.append(aux)

#     @classmethod
#     def pop_sum(cls, device=None, num_moe_layers=4):
#         """
#         弹出并求和。
#         增加 num_moe_layers 参数，默认 4 个 MoE 层。
#         """
#         if not cls._aux:
#             return None

#         # 💥 核心修复：只取当前最新前向传播生成的最后 4 个 loss！
#         # 完美扔掉 YOLO 初始化时留下的 "死节点(Dummy Loss)"
#         valid_aux = cls._aux[-num_moe_layers:]

#         if device is not None:
#             processed_tensors = [a.to(device) for a in valid_aux]
#         else:
#             processed_tensors = valid_aux

#         # 求和
#         total_aux = torch.stack(processed_tensors).sum()

#         # 清空列表，迎接下一个 Batch
#         cls._aux = []

#         return total_aux

class MoEAuxCollector:
    _aux = []
    # 🚨 增加最大容量限制 (假设网络最多有 4-8 个 MoE 层，设为 12 是绝对安全的)
    _max_capacity = 12

    @classmethod
    def add(cls, aux):
        if aux is not None:
            cls._aux.append(aux)

            # 🛡️ 终极防御机制：如果列表长度超出正常范围，说明发生了 "只前向、未反向" 的堆积
            # 立刻踢掉最早的节点，释放庞大的计算图内存！
            while len(cls._aux) > cls._max_capacity:
                old_aux = cls._aux.pop(0)
                del old_aux  # 彻底销毁

    @classmethod
    def pop_sum(cls, device=None, num_moe_layers=4):
        if not cls._aux:
            return None

        valid_aux = cls._aux[-num_moe_layers:]

        if device is not None:
            processed_tensors = [a.to(device) for a in valid_aux]
        else:
            processed_tensors = valid_aux

        total_aux = torch.stack(processed_tensors).sum()

        # 清空列表
        cls._aux.clear()

        return total_aux

    @classmethod
    def force_clear(cls):
        cls._aux.clear()