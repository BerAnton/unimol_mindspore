# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
""""""
from typing import List
from functools import lru_cache

import numpy as np
import mindspore as ms

from .lmdb_dataset import LMDBDataset
from .transforms import get_atom_vocab, smi2coords, sample_conformation, \
                        truncate_sample, remove_hydrogens, crop_sample, \
                        normalize_coordinates, mask_sample, get_edge_types, \
                        get_distance_matrix, get_cross_distance_matrix, \
                        add_special_tokens, SPECIAL_TOKENS, \
                        pad_1d_samples, pad_2d_samples, pad_coordinates_sample


class SMILESDataset:
    """
    The dataset for SMILES entries from dataset.
    """
    def __init__(self, lmdb_dataset_path: LMDBDataset):
        self.lmdb_dataset = LMDBDataset(lmdb_dataset_path)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        smiles = self.lmdb_dataset[idx]["smi"]
        return smiles        

    def __len__(self):
        return len(self.lmdb_dataset)


class MoleculeDataset:
    """
    This dataset must be enveloped in ms.GeneratorDataset(..., column_names=...)
    Column names must match __getitem__ return arguments.
    """
    def __init__(
        self,
        lmdb_dataset_path: LMDBDataset,
        atoms_vocab_path: str,
        is_train: bool,
        remove_hydrogen: str = None,
        max_atoms: int  = 512,
        prob_mask: float = 0.15,
        prob_unmask: float = 0.15,
        prob_random_token: float = 0.1,
        coords_noise_type: str = "uniform",
        coords_noise_coef: float = 3.0
    ):
        self.vocab = get_atom_vocab(atoms_vocab_path)
        self.lmdb_dataset = LMDBDataset(lmdb_dataset_path)
        self.is_train = is_train
        self.remove_hydrogen = remove_hydrogen
        self.max_atoms = max_atoms

        self.prob_mask = prob_mask
        self.prob_unmask = prob_unmask
        self.prob_random_token = prob_random_token
        self.coords_noise_type = coords_noise_type
        self.coords_noise_coef = coords_noise_coef

        self.token_vocab = self.vocab.vocab()
        # TODO: rework to special methods like Vocab.bos, Vocab.eos, Vocab.pad, Vocab.mask
        self.pad_idx = self.token_vocab['<PAD>']
        self.mask_idx = self.token_vocab['<MASK>']
        self.token_weights = self._tokens_weights()

        if coords_noise_type == "normal":
            self.noise_func = lambda num_mask: np.random.randn(num_mask, 3) * coords_noise_coef
        elif coords_noise_type == "uniform":
            self.noise_func = lambda num_mask: np.random.uniform(low=-coords_noise_coef, high=coords_noise_coef, size=(num_mask, 3))
        else:
            raise ValueError(f"Noise type can be normal or uniform, not {coords_noise_type}")

    @lru_cache(maxsize=32)
    def __getitem__(self, idx: int):
        sample = self.lmdb_dataset[idx]
        smi = sample["smi"]
        atoms = sample["atoms"]
        coordinates = sample["coordinates"]
        result = self._sample_preprocess(smi, atoms, coordinates)
        masked_atoms, masked_coords, masked_distance, edge_types = result[:4]  # network input
        target_atoms, coordinates, distance = result[4:]  # target values
        return masked_atoms, masked_coords, masked_distance, edge_types, \
               target_atoms, coordinates, distance        

    def __len__(self):
        return len(self.lmdb_dataset)

    def _sample_preprocess(self, smi, atoms, coordinates):
        """Returns preprocessed data samples.
        
        Order of return objects (N = atoms count):
            masked_atoms - tokenized masked atom list, shape (N + 2)
            masked_coordinates - normalized and noised atom coordinates, shape (N + 2, 3) or (N + 2, 2)
            masked_distances - pair-wise atom distances from noised coordinates, shape (N + 2, N + 2)
            edge_types - atom types, from masked tokens, shape(N + 2, 1)
            target_atoms - tokenized atom list, contains atom tokens, which was masked, shape (N + 2)
            coordinates - normalized atom coordinates, shape (N + 2, 3) or (N + 2, 2)
            distances - pair-wise atom distances from normalized coordinates, shape (N + 2, N + 2)
            smi - SMILES string for molecule in numpy array, shape ()
        """
        if self.is_train:
            conformer_2d = smi2coords(smi)
            coordinates.append(conformer_2d)
        coordinates_sample = sample_conformation(coordinates)
        atom_list, coordinates = truncate_sample(atoms, coordinates_sample)
        if self.remove_hydrogen:
            atom_list, coordinates = remove_hydrogens(atom_list, coordinates, self.remove_hydrogen)
        atom_list, coordinates = crop_sample(atom_list, coordinates, self.max_atoms)
        coordinates = normalize_coordinates(coordinates)
        tokenized_atoms = np.asarray(self.vocab.tokens_to_ids(atom_list))
        target_atoms, masked_atoms, masked_coords = mask_sample(tokenized_atoms, coordinates, self.token_vocab,
                                                                self.prob_mask, self.prob_unmask,
                                                                self.prob_random_token, 
                                                                self.noise_func, self.pad_idx, 
                                                                self.mask_idx, self.token_weights)
        target_atoms = add_special_tokens(target_atoms, self.pad_idx, self.pad_idx)
        masked_atoms = add_special_tokens(masked_atoms, self.token_vocab['<BOS>'], self.token_vocab['<EOS>'])
        masked_coords = add_special_tokens(masked_coords, 0.0, 0.0)
        masked_distance = get_distance_matrix(masked_coords)
        edge_types = get_edge_types(masked_atoms, len(self.token_vocab))
        coordinates = add_special_tokens(coordinates, 0.0, 0.0)
        distance = get_distance_matrix(coordinates)
                        
        return masked_atoms, masked_coords, masked_distance, edge_types, \
               target_atoms, coordinates, distance

    def _tokens_weights(self):
        special_tokens_idx = [self.token_vocab[token] for token in SPECIAL_TOKENS]
        token_weights = np.ones(len(self.token_vocab))
        token_weights[special_tokens_idx] = 0
        token_weights = token_weights / token_weights.sum()

        return token_weights


def molecule_collate_fn(masked_atoms, masked_coords, masked_distance, edge_types,
                       target_atoms, coordinates, distance, batch_info) -> List[ms.Tensor]:
    """Function for batch preprocessing.
    
    In this case, function is used for padding samples to create batch with same shapes
    for each net input and output.
    Intended use: as per_batch_map argument in GeneratorDataset.batch(...) function.
    All results are casted to Python list, because per_batch_map function must return 
    a list of numpy.ndarrays, which afterwards will be casted to mindspore.Tensors.
    """
    max_batch_length = max([sample.shape[0] for sample in masked_atoms])
    
    masked_atoms = list(pad_1d_samples(masked_atoms, 0, max_batch_length))
    masked_coords = list(pad_coordinates_sample(masked_coords, 0.0, max_batch_length))
    masked_distance  = list(pad_2d_samples(masked_distance, 0.0, max_batch_length))
    edge_types = list(pad_2d_samples(edge_types, 0, max_batch_length))

    target_atoms = list(pad_1d_samples(target_atoms, 0, max_batch_length))
    coordinates = list(pad_coordinates_sample(coordinates, 0.0, max_batch_length))
    distance  = list(pad_2d_samples(distance, 0.0, max_batch_length))
    
    return masked_atoms, masked_coords, masked_distance, edge_types, \
           target_atoms, coordinates, distance
