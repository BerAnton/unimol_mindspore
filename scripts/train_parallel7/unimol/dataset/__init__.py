from .ligands_dataset import MoleculeDataset, molecule_collate_fn


def create_dataset(config, is_train=True):
    if config.task == "molecule_pretrain":
        train_dataset = MoleculeDataset(
            lmdb_dataset_path=config.train_lmdb_dataset_path,
            atoms_vocab_path=config.atoms_vocab_path,
            is_train=is_train,
            remove_hydrogen=config.remove_hydrogen,
            max_atoms=config.max_seq_len,
            prob_mask=config.prob_mask,
            prob_unmask=config.prob_unmask,
            prob_random_token=config.prob_random_token,
            coords_noise_type=config.coords_noise_type,
            coords_noise_coef=config.coords_noise_coef   
        )
        
    return train_dataset