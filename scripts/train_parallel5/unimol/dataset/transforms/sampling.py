from typing import List

import numpy as np


def sample_conformation(conformations: List[np.array]):
    conf_count = len(conformations)
    conformations = np.array(conformations)
    sample_idx = np.random.randint(conf_count)
    return conformations[sample_idx]


def truncate_sample(atom_list:List[str], coordinates: np.array):
    min_len = min(len(atom_list), len(coordinates))
    atom_list = atom_list[:min_len]
    coordinates = coordinates[:min_len]

    return atom_list, coordinates


def crop_sample(atom_list:List[str], coordinates: np.array, max_atoms: int):
    if len(atom_list) > max_atoms:
        atom_idxs = np.random.choice(len(atom_list), max_atoms, replace=False)
        atom_list = np.array(atom_list)[atom_idxs]
        coordinates = coordinates[atom_idxs]
    return atom_list, coordinates


def add_special_tokens(sequence: np.array, begin_token, end_token):
    """Function to add special tokens to beginning and ending of sequence"""
    begin_token = np.expand_dims(np.full_like(sequence[0], begin_token), 0)
    end_token = np.expand_dims(np.full_like(sequence[0], end_token), 0)
    sequence = np.concatenate((begin_token, sequence), axis=0)
    sequence = np.concatenate((sequence, end_token), axis=0)
    return sequence