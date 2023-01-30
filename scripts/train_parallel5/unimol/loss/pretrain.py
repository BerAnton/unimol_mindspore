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

from .constants import Distance


class MoleculePretrainLoss(nn.LossBase):
    """Loss for molecule pretrain task. 
    
    Calculates sum of losses:
        - NLLLoss for atom token prediction.
        - smooth L1-loss for coordinate prediction.
        - smooth L1-loss for distance prediction.

    Args:
        beta (float): beta coef for smooth L1-loss.
    Returns:
        loss (float): loss for all three tasks.
        token_loss (float): loss for masked token classification.
        coords_loss (float): loss for coordinates prediction.
        distance_loss (float): loss for distance prediction.
    """
    def __init__(self, 
                 token_vocab,
                 token_loss_coef,
                 coords_loss_coef,
                 distance_loss_coef,
                 beta=1.0,
                 reduction='mean'):
        super(MoleculePretrainLoss, self).__init__(reduction=reduction)
        self.token_vocab = token_vocab
        self.pad_idx = token_vocab['<PAD>']

        self.log_softmax = ops.LogSoftmax(axis=-1)
        self.nll_loss = nn.NLLLoss(ignore_index=self.pad_idx, reduction=reduction)
        self.smooth_l1 = nn.SmoothL1Loss(beta=beta)

        self.token_loss_coef = token_loss_coef
        self.coords_loss_coef = coords_loss_coef
        self.distance_loss_coef = distance_loss_coef

        self.distance_mean = Distance.mean
        self.distance_std = Distance.std
            
    def construct(self, prediction, target):
        """token_logits, pred_coords, pred_distance, x_norm, pair_rep_norm  = prediction
        tokens, coordinates, distance = target"""
        token_logits, pred_coords, pred_distance, x_norm, pair_rep_norm = prediction
        tokens, coordinates, distance = target
        
        token_loss = self.nll_loss(
            self.log_softmax(token_logits).transpose(0, 2, 1),
            ops.cast(tokens, ms.int32)
        )
        coords_loss = self.smooth_l1(
            pred_coords.view(-1, 3),
            coordinates.view(-1, 3)
        )
        coords_loss = coords_loss.sum() / ops.size(coords_loss)
        distance_loss = self._calculate_distance_loss(
            pred_distance, distance
        )
        distance_loss = distance_loss.sum() / ops.size(distance_loss)
        loss = token_loss * self.token_loss_coef + \
               coords_loss * self.coords_loss_coef + \
               distance_loss * self.distance_loss_coef
        return loss

    def _calculate_distance_loss(self, pred, target):
        target = (target - self.distance_mean) / self.distance_std
        distance_loss = self.smooth_l1(
            pred.view(-1),
            target.view(-1)
        )
        return distance_loss
        


class PockerPretrainLoss(nn.LossBase):
    pass