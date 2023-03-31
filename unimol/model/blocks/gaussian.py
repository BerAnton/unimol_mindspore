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
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops


class GaussianLayer(nn.Cell):
    def __init__(self, K: int, edge_types: int):
        super(GaussianLayer, self).__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1)
        self.bias = nn.Embedding(edge_types, 1)

    def construct(self, X, edge_types):
        edge_types = ops.cast(edge_types, ms.int32)
        mul = self.mul(edge_types)
        bias = self.bias(edge_types)
        X = mul * ops.expand_dims(X, -1) + bias
        X = ops.broadcast_to(X, (X.shape[0], X.shape[1], -1, self.K))
        mean = self.means.embedding_table.view(-1)
        std = self.stds.embedding_table.view(-1) + 1e-5
        out = self._gaussian(X, mean, std)
        return out

    def _gaussian(self, x, mean, std):
        pi = ms.Tensor(3.14159, ms.float32)
        a = ops.pow((2 * pi), 0.5)
        return ops.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)
