import torch

class MoEAuxCollector:
    _aux = []

    @classmethod
    def add(cls, aux):
        if aux is None:
            return

        aux_clone = aux.clone().detach()
        aux_clone.requires_grad = aux.requires_grad
        cls._aux.append(aux_clone)

    @classmethod
    def pop_sum(cls, device=None):
        if not cls._aux:
            return None
        if device is None:
            device = cls._aux[0].device
        processed_tensors = []
        for a in cls._aux:
            # 每一步操作都clone，避免和_aux共享内存
            a_reshaped = a.reshape(()).clone()
            a_to = a_reshaped.to(device).clone()
            processed_tensors.append(a_to)

        # 修复3：sum后再clone，切断和processed_tensors的依赖
        s = torch.stack(processed_tensors).sum().clone()

        # 修复4：把clear改为“重新赋值空列表”（非原地操作）
        # cls._aux.clear()  # 原地操作，注释掉
        cls._aux = []  # 非原地操作，重新创建空列表
        return s