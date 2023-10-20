#
# Author:   
# Date: 04/25/2021
#
""" Replaced token detection model for representation learning
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from ...deberta import *

__all__ = ['LMMaskPredictionHead', 'ReplacedTokenDetectionModel']


class LMMaskPredictionHead(nn.Module):
    """ Replaced token prediction head
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act
        self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.classifier = nn.Linear(config.hidden_size, 1)

    def forward(self, hidden_states, input_ids, input_mask, mask_labels=None):
        # b x d
        ctx_states = hidden_states[:, 0, :]
        seq_states = self.LayerNorm(ctx_states.unsqueeze(-2) + hidden_states)
        seq_states = self.dense(seq_states)
        seq_states = self.transform_act_fn(seq_states)

        # b x max_len
        logits = self.classifier(seq_states).squeeze(-1)
        mask_loss = torch.tensor(0).to(logits).float()
        if mask_labels is not None:
            mask_logits = logits.view(-1)
            mask_labels = mask_labels.view(-1)
            _input_mask = input_mask.view(-1).to(mask_logits)
            input_idx = (_input_mask > 0).nonzero().view(-1)
            mask_labels = torch.gather(mask_labels.to(mask_logits), 0, input_idx)
            mask_loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
            mask_logits = torch.gather(mask_logits, 0, input_idx).float()
            mask_loss = mask_loss_fn(mask_logits, mask_labels)
        return logits, mask_labels, mask_loss


class ReplacedTokenDetectionModel(NNModule):
    """ RTD with DeBERTa
    """

    def __init__(self, config, *wargs, **kwargs):
        super().__init__(config)
        self.deberta = DeBERTa(config)

        self.max_relative_positions = getattr(config, 'max_relative_positions', -1)
        self.position_buckets = getattr(config, 'position_buckets', -1)
        if self.max_relative_positions < 1:
            self.max_relative_positions = config.max_position_embeddings
        self.mask_predictions = LMMaskPredictionHead(self.deberta.config)

        self.config = config
        pool_config = PoolConfig(self.config)
        output_dim = self.deberta.config.hidden_size
        self.pooler = ContextPooler(pool_config)
        output_dim = self.pooler.output_dim()

        self.apply(self.init_weights)

    def forward(self, input_ids, input_mask=None, lm_labels=None, position_ids=None, attention_mask=None, type_ids=None,
                dis_labels=None, labels=None):
        device = list(self.parameters())[0].device
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        if type_ids is not None:
            type_ids = type_ids.to(device)
        if dis_labels is not None:
            dis_labels = dis_labels.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        else:
            attention_mask = input_mask

        encoder_output = self.deberta(input_ids, input_mask, type_ids, output_all_encoded_layers=True,
                                      position_ids=position_ids)
        encoder_layers = encoder_output['hidden_states']
        ctx_layer = encoder_layers[-1]
        (mask_logits, mask_labels, mask_loss) = self.mask_predictions(encoder_layers[-1], input_ids, input_mask,
                                                                      dis_labels)

        pooled_output = self.pooler(encoder_layers[-1])
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        loss = torch.tensor(0).to(logits)
        if labels is not None:
            if self.num_labels == 1:
                # regression task
                loss_fn = torch.nn.MSELoss()
                logits = logits.view(-1).to(labels.dtype)
                loss = loss_fn(logits, labels.view(-1))
            elif labels.dim() == 1 or labels.size(-1) == 1:
                label_index = (labels >= 0).nonzero()
                labels = labels.long()
                if label_index.size(0) > 0:
                    labeled_logits = torch.gather(logits, 0, label_index.expand(label_index.size(0), logits.size(1)))
                    labels = torch.gather(labels, 0, label_index.view(-1))
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(labeled_logits.view(-1, self.num_labels).float(), labels.view(-1))
                else:
                    loss = torch.tensor(0).to(logits)
            else:
                log_softmax = torch.nn.LogSoftmax(-1)
                label_confidence = 1
                loss = -((log_softmax(logits) * labels).sum(-1) * label_confidence).mean()
        return {
            'logits': logits,
            'loss': loss
        }
