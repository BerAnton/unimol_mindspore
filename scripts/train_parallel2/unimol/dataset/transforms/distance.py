import numpy as np
from scipy.spatial import distance_matrix


def get_distance_matrix(coordinates: np.array):
    coordinates = coordinates.reshape(-1, 3)
    distance = distance_matrix(coordinates, coordinates).astype(np.float32)
    return distance


def get_cross_distance_matrix(molecule_coords: np.array, pocket_coords: np.array):
    molecule_coords = molecule_coords.reshape(-1, 3)
    pocket_coords = pocket_coords.reshape(-1, 3)
    distance = distance_matrix(molecule_coords, pocket_coords).astype(np.float32)
    return distance
