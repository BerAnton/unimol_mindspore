import numpy as np


def get_edge_types(tokens: np.array, types_count: int):
    offset = tokens.reshape(-1, 1) * types_count + tokens.reshape(-1, 1)
    return offset