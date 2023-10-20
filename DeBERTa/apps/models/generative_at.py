#     
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Author:   
# Date: 01/25/2019
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from ..models import MaskedLanguageModel
from ..models import SequenceClassificationModel

from ...deberta import *
import random

__all__ = ['GenerativeAdversarialTrainingModel']


class GenerativeAdversarialTrainingModel(NNModule):
    def __init__(self, config, num_labels=2, drop_out=None, pre_trained=None):
        super().__init__(config)
        gen_config = config.generator
        disc_config = config.discriminator
        self.num_labels = num_labels

        self.discriminator = SequenceClassificationModel(disc_config, num_labels=num_labels)
        self.generator = MaskedLanguageModel(gen_config)

        self.discriminator._register_load_state_dict_pre_hook(self._pre_load_hook)
        self.generator._register_load_state_dict_pre_hook(self._pre_load_hook)  # 改名

        self.config = self.discriminator.config

        self.apply(self.init_weights)


    def forward(self, input_ids, input_mask=None, lm_labels=None, position_ids=None, attention_mask=None, type_ids=None,
                dis_labels=None, labels=None, **kwargs):
        device = list(self.parameters())[0].device
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        if type_ids is not None:
            type_ids = type_ids.to(device)
        if dis_labels is not None:
            dis_labels = dis_labels.to(device)

        outputs = self.discriminator(input_ids, type_ids, input_mask, labels, position_ids, **kwargs)
        encoder_layers = outputs['hidden_states']
        (mask_logits, mask_labels, mask_loss) = self.discriminator.mask_predictions(encoder_layers[-1], input_ids, input_mask,
                                                                      dis_labels)
        return {
            'logits': outputs['logits'],
            'loss': outputs['loss'],
            'mask_loss': mask_loss,
            'mask_logits': mask_logits,
            'mask_labels': mask_labels
        }

    def export_onnx(self, onnx_path, input):
        del input[0]['labels']  # = input[0]['labels'].unsqueeze(1)
        torch.onnx.export(self, input, onnx_path, opset_version=13, do_constant_folding=False, \
                          input_names=['input_ids', 'type_ids', 'input_mask', 'position_ids', 'labels'],
                          output_names=['logits', 'loss'], \
                          dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence_length'}, \
                                        'type_ids': {0: 'batch_size', 1: 'sequence_length'}, \
                                        'input_mask': {0: 'batch_size', 1: 'sequence_length'}, \
                                        'position_ids': {0: 'batch_size', 1: 'sequence_length'}, \
                                        #     'labels' : {0 : 'batch_size', 1: 'sequence_length'}, \
                                        })

    def _pre_load_hook(self, state_dict, prefix, local_metadata, strict,
                       missing_keys, unexpected_keys, error_msgs):
        new_state = dict()
        bert_prefix = prefix + 'bert.'
        deberta_prefix = prefix + 'deberta.'
        for k in list(state_dict.keys()):
            if k.startswith(bert_prefix):
                nk = deberta_prefix + k[len(bert_prefix):]
                value = state_dict[k]
                del state_dict[k]
                state_dict[nk] = value

    @staticmethod
    def _set_param(module, param_name, value):
        if hasattr(module, param_name):
            delattr(module, param_name)
        module.register_buffer(param_name, value)

    def register_discriminator_fw_hook(self, *wargs):
        def fw_hook(module, *inputs):
            g_w_ebd = self.generator.deberta.embeddings.word_embeddings
            d_w_ebd = self.discriminator.deberta.embeddings.word_embeddings
            self._set_param(g_w_ebd, 'weight', d_w_ebd.weight)

            g_p_ebd = self.generator.deberta.embeddings.position_embeddings
            d_p_ebd = self.discriminator.deberta.embeddings.position_embeddings
            self._set_param(g_p_ebd, 'weight', d_p_ebd.weight)
            return None

        self.discriminator.register_forward_pre_hook(fw_hook)

    def make_electra_data(self, input_data, temp=1, rand=None):
        new_data = input_data.copy()
        if rand is None:
            rand = random
        new_data['input_ids'] = new_data['masked_token_ids']
        gen = self.generator_fw(**new_data)
        lm_logits = gen['logits']
        lm_labels = input_data['lm_labels']
        lm_loss = gen['loss']
        mask_index = (lm_labels.view(-1) > 0).nonzero().view(-1)
        gen_pred = torch.argmax(lm_logits, dim=1).detach().cpu().numpy()
        topk_labels, top_p = self.topk_sampling(lm_logits, topk=1, temp=temp)

        top_ids = torch.zeros_like(lm_labels.view(-1))
        top_ids.scatter_(index=mask_index, src=topk_labels.view(-1).int(), dim=-1)
        top_ids = top_ids.view(lm_labels.size())
        new_ids = torch.where(lm_labels > 0, top_ids, input_data['input_ids'])
        new_data['input_ids'] = new_ids.detach()
        return new_data, lm_loss, gen

    def generator_fw(self, **kwargs):
        return self.generator(**kwargs)

    def topk_sampling(self, logits, topk=1, start=0, temp=1):
        top_p = torch.nn.functional.softmax(logits / temp, dim=-1)
        topk = max(1, topk)
        next_tokens = torch.multinomial(top_p, topk)
        return next_tokens, top_p