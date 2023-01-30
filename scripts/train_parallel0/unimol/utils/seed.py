import numpy as np
import mindspore as ms


def seed_all(seed: int):
    np.random.seed(23)
    ms.set_seed(23)
