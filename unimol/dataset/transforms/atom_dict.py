from mindspore.dataset.text import Vocab


# PAD token must be first
SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<MASK>"] 


def get_atom_vocab(filepath: str):
    vocab = Vocab().from_file(filepath, special_tokens=SPECIAL_TOKENS)
    return vocab
