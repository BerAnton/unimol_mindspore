import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

from .attention import MultiHeadSelfAttention


class EncoderLayer(nn.Cell):

    def __init__(
        self,
        emb_dim: int,
        ff_dim: int,
        attention_heads: int,
        dropout: float
    ):
        super(EncoderLayer, self).__init__()

        self.emb_dim = emb_dim
        self.attention_heads = attention_heads
        self.attention = MultiHeadSelfAttention(
            self.emb_dim, self.attention_heads, dropout
        )
        self.linear = nn.SequentialCell([
            nn.Dense(self.emb_dim, ff_dim),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Dense(ff_dim, self.emb_dim)
        ])
        self.dropout = nn.Dropout(dropout)
        self.attention_layer_norm = nn.LayerNorm([self.emb_dim])
        self.post_layer_norm = nn.LayerNorm([self.emb_dim])
        
        
    def construct(
        self,
        X: ms.Tensor,
        attention_bias: ms.Tensor = None,
        padding_mask: ms.Tensor = None,
        return_attention: bool = False
    ):
        attn_out = self.attention(
            X,
            padding_mask,
            attention_bias,
            return_attention
        )
        if return_attention:
            out, attention_out = attn_out
        else:
            attention_out = None
            out = attn_out
        X = X + self.dropout(out)
        X = self.attention_layer_norm(X)

        ff_out = self.linear(X)
        X = X + self.dropout(ff_out)
        X = self.post_layer_norm(X)
        if return_attention:
            return X, attention_out
        return X
