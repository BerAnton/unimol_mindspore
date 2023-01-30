from typing import List

from mindspore import Tensor
import numpy as np


def pad_1d_samples(samples: List[np.ndarray], pad_token: int, length_to_pad: int) -> np.ndarray:
    """Function for padding sequence 1D-batch to longest sequence in batch
    """
    padded_samples = np.full((len(samples), length_to_pad), pad_token).astype(samples[0].dtype)
    for i, sample in enumerate(samples):
        padded_samples[i][:sample.shape[0]] = sample.copy()
    return padded_samples


def pad_2d_samples(samples: List[np.ndarray], pad_token: int, length_to_pad: int):
    """Function for padding matrices in batch to biggest matrices in batch"""
    padded_samples = np.full((len(samples), length_to_pad, length_to_pad), pad_token).astype(samples[0].dtype)
    for i, sample in enumerate(samples):
        padded_samples[i][:sample.shape[0], :sample.shape[0]] = sample.copy()
    return padded_samples


def pad_coordinates_sample(coordinates_samples: List[np.ndarray], pad_token: int, length_to_pad: int):
    padded_coordinates = np.full((len(coordinates_samples), length_to_pad, 3), pad_token).astype(coordinates_samples[0].dtype)
    for i, sample in enumerate(coordinates_samples):
        padded_coordinates[i][:sample.shape[0], :] = sample.copy()
    return padded_coordinates
