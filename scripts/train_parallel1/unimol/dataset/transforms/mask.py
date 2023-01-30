from typing import List, Callable

import numpy as np
from mindspore.dataset.text import Vocab


def mask_sample(
    atom_list: List[str],
    coordinates: np.array,
    dictionary: Vocab,
    prob_mask: float,
    prob_unmask: float,
    prob_random_token: float,
    coords_noise_func: Callable,
    pad_idx: int,
    mask_idx: int,
    tokens_weights: np.array
):
    """Function for masking atoms and noising atoms' coordinates for training.
    Args:
        atom_list: sample atom list.
        coordinates: atoms' coordinates array with shape (len(atom_list), 3).
        dictionary: mindspore Vocab, which was used for atoms tokenization.
        prob_mask: probability for atom masking.
        prob_unmask: probability for unmasking atom, which was replace with <MASK> token.
        prob_random_token: probability for token replacement.
        coords_noise_type: name of distribution for sampling coordinates noise. Could be "uniform" or "normal"
        coords_noise_coef: noise multiplier for normal noise, upper bound for uniform noise.
    Returns:
        target_atom_list (np.array): masked atoms answer, others replaced by <PAD>.
        masked_atom_list (np.array): masked atoms list.
        masked_coordinates (np.array): coordinates array with noise added to masked atoms' coordinates.

    """
    replace_unmask = (prob_unmask + prob_random_token) > 0

    # mask and target sequence generation
    size = len(atom_list)
    mask_size = int(prob_mask * size + np.random.rand())
    mask_idxs = np.random.choice(size, mask_size, replace=False)
    mask = np.full(size, False)
    mask[mask_idxs] = True
    target_atom_list = np.full(len(mask), pad_idx)
    target_atom_list[mask] = atom_list[mask]

    # token replacement or unmasking masks 
    if replace_unmask:
        prob = prob_unmask + prob_random_token
        additional_mask = mask & (np.random.rand(size) < prob)
        if prob_random_token == 0.0:
            unmask_mask = additional_mask
            replacement_mask = None
        elif prob_unmask == 0.0:
            replacement_mask = additional_mask
            unmask_mask = None
        else:
            prob_unmask = prob_unmask / prob
            unmask_mask = additional_mask & (np.random.rand(size) < prob_unmask)
            replacement_mask = additional_mask & (~(np.random.rand(size) < prob_unmask))
        if unmask_mask is not None:
            mask = mask ^ unmask_mask
    
    masked_atom_list = np.copy(atom_list)
    masked_atom_list[mask] = mask_idx

    if replacement_mask is not None:
        replacement_counts = replacement_mask.sum()
        if replacement_counts > 0:
            masked_atom_list[replacement_mask] = np.random.choice(len(dictionary),
                                                                  replacement_counts,
                                                                  p=tokens_weights)
    masked_coordinates = np.copy(coordinates)
    masked_coordinates[mask, :] += coords_noise_func(mask.astype(np.int32).sum())
    
    return target_atom_list, masked_atom_list, masked_coordinates
