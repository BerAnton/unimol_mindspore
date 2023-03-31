import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops


class MultiHeadSelfAttention(nn.Cell):
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout_rate: float = None,
    ):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.head_dim = embed_dim // num_heads
        if self.embed_dim % num_heads != 0:
            raise ValueError(
                f"Head dimension {self.head_dim} is not divisible by number of heads - {num_heads}"
            )

        self.scaling_factor = ops.pow(ops.sqrt(ms.Tensor(self.head_dim, ms.float32)), -0.5)
        self.fill_value = ms.Tensor(float("-inf"), dtype=ms.float32)

        if self.dropout_rate is not None:
            self.dropout = nn.Dropout(dropout_rate)
          
        # concat Q, K, V for efficiency
        self.qkv_proj = nn.Dense(embed_dim, embed_dim * 3)
        self.out_proj = nn.Dense(embed_dim, embed_dim)
        self.softmax = nn.Softmax(axis=-1)
                        
    def construct(
        self,
        query: ms.Tensor,
        padding_mask: ms.Tensor = None,
        attention_bias: ms.Tensor = None,
        return_attention: bool = False,
    ):

        batch_size, target_len, emb_dim = query.shape

        QKV = self.qkv_proj(query)
        QKV = ops.reshape(QKV, (batch_size, target_len,
                           self.num_heads, self.head_dim * 3))
        QKV = ops.transpose(QKV, (0, 2, 1, 3))
        Q, K, V = ops.split(QKV, axis=-1, output_num=3)
        V = V.view(batch_size * self.num_heads, -1, self.head_dim)

        source_len = K.shape[2]

        attention_logits = ops.matmul(Q, ops.transpose(K, (0, 1, 3, 2)))
        attention_logits = attention_logits * self.scaling_factor

        if padding_mask is not None:
            attention_logits = ops.reshape(attention_logits, (batch_size, self.num_heads, target_len, source_len))
            padding_mask = ops.cast(padding_mask, ms.bool_)
            attention_logits = ops.masked_fill(attention_logits, padding_mask, self.fill_value)
        attention_logits = attention_logits.view(batch_size * self.num_heads, target_len, source_len)
        # if we return attention we need to add attention bias, which was passed as arg
        # attention bias in UniMol case is pair representation tensor
        if return_attention:
            attention_logits += attention_bias
            attention = self.softmax(attention_logits)
        else:
            attention = self.softmax(attention_logits)
        if self.dropout_rate:
            attention = self.dropout(attention)
        
        output = ops.matmul(attention, V)
        output = ops.transpose(output.view(batch_size, self.num_heads, target_len, self.head_dim),
                               (0, 2, 1, 3))
        output = ops.reshape(output, (batch_size, target_len, emb_dim))
        output = self.out_proj(output)

        if return_attention:
            return output, attention
        return output
