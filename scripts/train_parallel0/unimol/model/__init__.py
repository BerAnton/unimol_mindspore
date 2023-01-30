from .unimol import UniMol, UnimolWithLoss


def create_model(atom_dict, config, loss_fn):
    net = UniMol(
        dictionary=atom_dict,
        encoder_layers=config.encoder_layers,
        encoder_emb_dim=config.encoder_emb_dim,
        encoder_ff_emb_dim=config.encoder_ff_emb_dim,
        encoder_attention_heads=config.encoder_attention_heads,
        gaus_kernel_channels=config.gaus_kernel_channels,
        dropout=config.dropout,
        max_seq_len=config.max_seq_len,
    )
    net_with_loss = UnimolWithLoss(net, loss_fn)
    return net_with_loss