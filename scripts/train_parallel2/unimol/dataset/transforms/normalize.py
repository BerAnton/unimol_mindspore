import numpy as np


def normalize_coordinates(coordinates: np.array):
    coordinates = coordinates - coordinates.mean(axis=0)
    return coordinates
