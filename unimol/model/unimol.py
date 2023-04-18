from typing import Dict

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

from .blocks.transformer_encoder_pair import EncoderWithPair
from .blocks.gaussian import GaussianLayer
from .blocks.heads import MaskHead, ClassifictaionHead, NonLinearHead, DistanceHead


class UniMol(nn.Cell):
    """
    Attention note:
        In Mindspore, nn.Dropout keep_prob stands for not zeroing probability,
        unlike PyTorch.  
    """
    def __init__(
        self,
        dictionary: Dict,
        encoder_layers: int,
        encoder_emb_dim: int,
        encoder_ff_emb_dim: int,
        encoder_attention_heads: int,
        gaus_kernel_channels: int,
        dropout: float,
        max_seq_len: int,
    ):
        super(UniMol, self).__init__()
        self.padding_idx = dictionary['<PAD>']
        self.encoder_emb_dim = encoder_emb_dim
        self.token_emb = nn.Embedding(
            len(dictionary), encoder_emb_dim, 
            padding_idx=self.padding_idx
        )
        self.encoder = EncoderWithPair(
            encoder_layers=encoder_layers,
            emb_dim=encoder_emb_dim,
            ff_dim=encoder_ff_emb_dim,
            attention_heads=encoder_attention_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )

        self.classification_heads = nn.CellList()

        self.mask_head = MaskHead(
            emb_dim=encoder_emb_dim,
            out_dim=len(dictionary)
        )
        
        self.pair_coord_proj = NonLinearHead(
            encoder_attention_heads, encoder_attention_heads, 1
        )
        
        self.distance_head = DistanceHead(encoder_attention_heads)
        
        self.K = gaus_kernel_channels
        self.edge_types_count = len(dictionary) * len(dictionary)
        self.gaussian_proj = NonLinearHead(
            self.K, self.K, encoder_attention_heads
        )
        self.gaussian = GaussianLayer(
            self.K, self.edge_types_count
        )

    def construct(
        self,
        tokens: ms.Tensor,
        coords: ms.Tensor,
        distance: ms.Tensor,
        edge_types: ms.Tensor,
        masked_tokens=None,
        features_only=False
    ):
        pad_mask = ops.cast(ops.equal(tokens, self.padding_idx), coords.dtype)
        embed_tokens = self.token_emb(tokens)
        graph_attn_bias = self._get_distance_features(distance, edge_types)
        (encoder_rep, encoder_pair_rep, delta_encoder_pair_rep,
              x_norm, delta_encoder_pair_rep_norm) = self.encoder(embed_tokens, graph_attn_bias, pad_mask)
        encoder_pair_rep[ops.equal(encoder_pair_rep, float("-inf"))] = 0  # unsupported syntax in graph mode
        encoder_distance = None
        encoder_coords = None
        if features_only:
            return (encoder_rep, encoder_coords, encoder_distance, x_norm, delta_encoder_pair_rep_norm)
        logits = self.mask_head(encoder_rep, masked_tokens)
        coords_emb = coords
        if pad_mask.sum():  # Maybe error somewhere here
            atom_num = ((ops.ones_like(pad_mask) - pad_mask).sum(axis=1) - 1).view(-1, 1, 1, 1)
        else:
            shape = ops.shape(coords)
            atom_num = ms.Tensor(shape[1] - 1, dtype=ms.int32)  # unsupported syntax in graph mode
        delta_coords = ops.expand_dims(coords_emb, axis=1) - ops.expand_dims(coords_emb, axis=2)
        attention_probs = self.pair_coord_proj(delta_encoder_pair_rep)
        coords_update = delta_coords / atom_num * attention_probs
        coords_update = coords_update.sum(axis=2)
        encoder_coords = coords_emb + coords_update

        encoder_distance = self.distance_head(encoder_pair_rep)
                
        if self.training:
            return (logits, encoder_coords, encoder_distance, x_norm, delta_encoder_pair_rep_norm)
        else:
            return (encoder_rep, encoder_pair_rep)
   
    def _get_distance_features(self, distances, edge_types):
        nodes_count = distances.shape[-1]
        gaus_features = self.gaussian(distances, edge_types)
        result = self.gaussian_proj(gaus_features)
        graph_attention_bias = ops.transpose(result, (0, 3, 1, 2)).view(-1, nodes_count, nodes_count)
        return graph_attention_bias


class UnimolWithLoss(nn.Cell):
    
    def __init__(self, network, loss_fn):
        super(UnimolWithLoss, self).__init__()
        self.network = network
        self.loss_fn = loss_fn

    def construct(self, masked_atoms, masked_coords, masked_distance, 
                  edge_types, target_atoms, coordinates, distance):
        target = target_atoms, coordinates, distance
        prediction = self.network(masked_atoms, masked_coords,
                                  masked_distance, edge_types)
        loss = self.loss_fn(prediction, target)
        return loss


class UniMolEval(nn.Cell):

    def __init__(self, network, metrics):
        super(UniMolEval, self).__init__()
        self.network = network
        self.metrics = metrics

    def construct(self, masked_atoms, masked_coords, masked_distance, 
                  edge_types, target_atoms, coordinates, distance):
        pass