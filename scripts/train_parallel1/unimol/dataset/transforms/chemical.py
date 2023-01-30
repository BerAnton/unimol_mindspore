from typing import List

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def smi2coords(smiles: str):
    """Generates 2D coordinates from molecule's SMILES
    Args:
        smiles (str): molecule's SMILES representation.
    Returns:
        coordinates (List[float]): list of atoms coordinates in molecule.
    """

    molecule = Chem.MolFromSmiles(smiles)
    molecule = AllChem.AddHs(molecule)
    AllChem.Compute2DCoords(molecule)
    coordinates = molecule.GetConformer().GetPositions().astype(np.float32)

    return coordinates


# TODO: maybe remove mode == "polar"?
def remove_hydrogens(atom_list: List[str], coordinates: List[np.array], mode: str):
    """Function for removing hydrogens from atom and coordinates representation
    Args:
        mode (str): "all" or "polar".
    """
    if mode == "all":
        other_atoms_idxs = atom_list != "H"
        atom_list = atom_list[other_atoms_idxs]
        coordinates = coordinates[other_atoms_idxs]
    elif mode == "polar":
        idx = 0
        for i, atom in enumerate(atom_list[::-1]):
            if atom != "H":
                break
            else:
                idx = i + 1
        if idx != 0:
            atom_list = atom_list[:-idx]
            coordinates = coordinates[:-idx]
    return atom_list, coordinates

