import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops


class MaskHead(nn.Cell):
    """Head for masked language modelling"""

    def __init__(self, emb_dim: int, out_dim: int):
        super(MaskHead, self).__init__()
        self.dense = nn.Dense(emb_dim, emb_dim)
        self.activation = nn.GELU()
        self.layernorm = nn.LayerNorm([emb_dim])
        self.out_dense = nn.Dense(emb_dim, out_dim)
            
    def construct(self, features, mask_tokens=None):
        X = self.dense(features)
        X = self.activation(X)
        X = self.layernorm(X)
        X = self.out_dense(X)
        return X
    

class ClassifictaionHead(nn.Cell):
    
    def __init__(
            self,
            input_dim: int,
            hid_dim: int, 
            num_classes: int, 
            dropout: float
        ):
        super(ClassifictaionHead, self).__init__()
        self.dense = nn.Dense(input_dim, hid_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.out_dense = nn.Dense(hid_dim, num_classes)
    
    def construct(self, features):
        X = features[:, 0, :]
        X = self.dropout(X)
        X = self.dense(X)
        X = self.activation(X)
        X = self.dropout(X)
        X = self.out_dense(X)
        return X


class NonLinearHead(nn.Cell):

    def __init__(
            self,
            input_dim: int,
            hid_dim: int,
            out_dim: int            
        ):
        super(NonLinearHead, self).__init__()
        self.in_dense = nn.Dense(input_dim, hid_dim)
        self.out_dense = nn.Dense(hid_dim, out_dim)
        self.activation = nn.GELU()
    
    def construct(self, X):
        X = self.in_dense(X)
        X = self.activation(X)
        X = self.out_dense(X)
        return X


class DistanceHead(nn.Cell):

    def __init__(self, heads):
        super(DistanceHead, self).__init__()
        self.dense = nn.Dense(heads, heads)
        self.layernorm = nn.LayerNorm([heads])
        self.out_dense = nn.Dense(heads, 1)
        self.activation = nn.GELU()
    
    def construct(self, X):
        batch_size, seq_length, seq_length, _ = X.shape
        X = self.activation(self.dense(X))
        X = self.layernorm(X)
        X = self.out_dense(X).view(batch_size, seq_length, seq_length)
        X = (X + ops.transpose(X, (0, 2, 1))) * 0.5
        return X
