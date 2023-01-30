import mindspore as ms
from dataclasses import dataclass


class Distance:
    mean: ms.Tensor = ms.Tensor(6.312581655060595, dtype=ms.float32)
    std: ms.Tensor = ms.Tensor(3.3899264663911888, dtype=ms.float32)