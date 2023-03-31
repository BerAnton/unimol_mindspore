import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

from .transformer_encoder_layer import EncoderLayer


class EncoderWithPair(nn.Cell):

    def __init__(
        self,
        encoder_layers: int,
        emb_dim: int,
        ff_dim: int,
        attention_heads: int,
        dropout: int,
        max_seq_len: int
    ):
        super(EncoderWithPair, self).__init__()
        self.emb_dim = emb_dim
        self.max_seq_len = max_seq_len
        self.emb_layernorm = nn.LayerNorm([self.emb_dim])
        self.final_norm = nn.LayerNorm([self.emb_dim])
        self.head_layernorm = nn.LayerNorm([attention_heads])
                
        self.layers = nn.CellList([
            EncoderLayer(
                emb_dim=self.emb_dim,
                ff_dim=ff_dim,
                attention_heads=attention_heads,
                dropout=dropout,
            )
            for _ in range(encoder_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        
        self.fill_value = ms.Tensor(0, dtype=ms.float32)
        
    def construct(
        self,
        embedding: ms.Tensor,
        attention_bias: ms.Tensor = None,
        padding_mask: ms.Tensor = None
    ):
        batch_size, seq_length = embedding.shape[0:2]
        X = self.emb_layernorm(embedding)
        X = self.dropout(X)
        if padding_mask is not None:
            X = X * (1 - ops.expand_dims(padding_mask, -1))
        input_attention_bias = attention_bias
        input_padding_mask = padding_mask
        attention_bias, padding_mask = self._fill_attn_bias(batch_size, seq_length, attention_bias, padding_mask)

        for i in range(len(self.layers)):
            X, attention_bias = self.layers[i](
                X, attention_bias, padding_mask, return_attention=True
            )
        
        X_norm = self._norm_loss(X)
        if input_padding_mask is not None:
            token_mask = 1.0 - input_padding_mask
        else:
            token_mask = ops.ones_like(X_norm, device=X.device)
        X_norm = self._masked_mean(token_mask, X_norm)

        X = self.final_norm(X)
        
        delta_pair_repr = attention_bias - input_attention_bias
        delta_pair_repr, _ = self._fill_attn_bias(batch_size, seq_length, delta_pair_repr, input_padding_mask)
        attention_bias = ops.transpose(
            attention_bias.view(batch_size, -1, seq_length, seq_length),
            (0, 2, 3, 1)
        )
        delta_pair_repr = ops.transpose(
            delta_pair_repr.view(batch_size, -1, seq_length, seq_length),
            (0, 2, 3, 1)
        )
        pair_mask = token_mask[..., None] * token_mask[..., None, :]
        delta_pair_repr_norm = self._norm_loss(delta_pair_repr)
        delta_pair_repr_norm = self._masked_mean(pair_mask, delta_pair_repr_norm, axis=(-1, -2))

        delta_pair_repr = self.head_layernorm(delta_pair_repr)

        return X, attention_bias, delta_pair_repr, X_norm, delta_pair_repr_norm


    def _fill_attn_bias(self, batch_size, seq_length, 
                        attention_bias,
                        padding_mask):  
        """Function to merge attention bias and padding mask."""
        attention_bias = attention_bias.view(
            batch_size, -1, seq_length, seq_length
        )
        padding_mask = ops.expand_dims(ops.expand_dims(padding_mask, 1), 2)
        padding_mask = ops.cast(padding_mask, ms.bool_)
        attention_bias = ops.masked_fill(attention_bias, padding_mask,
                                   self.fill_value).view(-1, seq_length, seq_length)
        return attention_bias, padding_mask

    def _norm_loss(self, x, eps=1e-10, tolerance=1.0):
        """Function to calculate additional norm loss for model. 
        
        See paper Appendix F for info.
        """
        max_norm = x.shape[-1] ** 0.5
        squares = ops.pow(x, 2)
        norm = ops.sqrt(squares.sum(axis=-1) + eps)
        error = ops.ReLU()((norm - max_norm).abs() - tolerance)
        return error
    
    @staticmethod
    def _masked_mean(mask, value, axis=-1, eps=1e-10):
        return (mask * value).sum(axis=axis) / (eps + mask.sum(axis=axis)).mean()
