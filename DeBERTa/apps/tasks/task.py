#
# Author:   
# Date: 01/25/2019
#

import csv
import json
import os
from collections import OrderedDict

from .metrics import *
from ..models import SequenceClassificationModel
from ..models import GenerativeAdversarialTrainingModel
from ...utils import get_logger
import torch
from ...data.example import _truncate_segments
import random

logger = get_logger()

__all__ = ['EvalData', 'Task']


class EvalData:
    def __init__(self, name, examples, metrics_fn=None, predict_fn=None, ignore_metric=False, critial_metrics=None):
        def accuracy_fn(logits, labels):
            return OrderedDict(accuracy=metric_accuracy(logits, labels))

        def default_pred_fn(logits, output_dir, name, prefix):
            output = os.path.join(output_dir, 'submit-{}-{}.tsv'.format(name, prefix))
            preds = np.argmax(logits, axis=-1)
            with open(output, 'w', encoding='utf-8') as fs:
                fs.write('index\tpredictions\n')
                for i, p in enumerate(preds):
                    fs.write('{}\t{}\n'.format(i, p))

        self.name = name
        self.data = examples
        self.ignore_metric = ignore_metric
        self.critial_metrics = critial_metrics
        self.metrics_fn = metrics_fn if metrics_fn is not None else accuracy_fn
        self.predict_fn = predict_fn if predict_fn is not None else default_pred_fn

    def __repr__(self):
        return f'{self.name}, {type(self.data)}: {len(self.data)}, {self.predict_fn}, {self.metrics_fn}'


class Task():
    _meta = {}

    def __init__(self, tokenizer, args, **kwargs):
        self.tokenizer = tokenizer
        self.args = args

    def eval_data(self, **kwargs):
        raise NotImplementedError('Eval_data method not implemented yet.')

    def train_data(self, **kwargs):
        raise NotImplementedError('Eval_data method not implemented yet.')

    def test_data(self, **kwargs):
        raise NotImplementedError('Eval_data method not implemented yet.')

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def label2id(self, labelstr):
        label_dict = {l: i for i, l in enumerate(self.get_labels())}
        return label_dict[labelstr] if labelstr in label_dict else -1

    def get_train_fn(self, *args, **kwargs):
        return None

    def get_eval_fn(self, *args, **kwargs):
        return None

    def get_pred_fn(self, *args, **kwargs):
        return None

    def get_loss_fn(self, *args, **kwargs):
        return None

    def get_metrics_fn(self):
        """Calcuate metrics based on prediction results"""

        def metrics_fn(logits, labels):
            return OrderedDict(accuracy=metric_accuracy(logits, labels))

        return metrics_fn

    def get_predict_fn(self):
        """Calcuate metrics based on prediction results"""

        def predict_fn(logits, output_dir, name, prefix):
            output = os.path.join(output_dir, 'submit-{}-{}.tsv'.format(name, prefix))
            preds = np.argmax(logits, axis=-1)
            labels = self.get_labels()
            with open(output, 'w', encoding='utf-8') as fs:
                fs.write('index\tpredictions\n')
                for i, p in enumerate(preds):
                    fs.write('{}\t{}\n'.format(i, labels[p]))

        return predict_fn

    def get_adv_predict_fn(self):
        """Calcuate metrics based on prediction results"""

        def predict_fn(logits):
            preds = np.argmax(logits, axis=-1)
            # labels = self.get_labels()
            return [int(p) for p in preds]

        return predict_fn

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    def get_feature_fn(self, max_seq_len=512, mask_gen=None):
        def _example_to_feature(example, rng=None, ext_params=None, **kwargs):
            return self.example_to_feature(self.tokenizer, example, max_seq_len=max_seq_len, \
                                           rng=rng, mask_generator=mask_gen, ext_params=ext_params, **kwargs)

        return _example_to_feature

    def example_to_feature(self, tokenizer, example, max_seq_len=512, rng=None, mask_generator=None,
                           ext_params=None,
                           **kwargs):
        if not rng:
            rng = random
        max_num_tokens = max_seq_len - len(example.segments) - 1 - 1
        segments = _truncate_segments([tokenizer.tokenize(s) for s in example.segments], max_num_tokens, rng)

        _tokens = ['[CLS]']
        type_ids = [0]
        for i, s in enumerate(segments):
            _tokens.extend(s)
            _tokens.append('[SEP]')
            type_ids.extend([(i + 1) % 2] * (len(s) + 1))
        if mask_generator:
            token_ids = tokenizer.convert_tokens_to_ids(_tokens)
            tokens, lm_labels = mask_generator.mask_tokens(_tokens, rng)

            dis_labels = [0] * len(lm_labels)
            for i, s in enumerate(lm_labels):
                if s != 0:
                    dis_labels[i] = 1
            masked_token_ids = tokenizer.convert_tokens_to_ids(tokens)
            features = OrderedDict(input_ids=token_ids,
                                   masked_token_ids=masked_token_ids,
                                   type_ids=type_ids,
                                   position_ids=list(range(len(token_ids))),
                                   input_mask=[1] * len(token_ids),
                                   dis_labels=dis_labels,
                                   lm_labels=lm_labels)
        else:
            dis_labels = [0] * len(_tokens)

            token_ids = tokenizer.convert_tokens_to_ids(_tokens)
            features = OrderedDict(input_ids=token_ids,
                                   type_ids=type_ids,
                                   position_ids=list(range(len(token_ids))),
                                   input_mask=[1] * len(token_ids),
                                   dis_labels=dis_labels)

        for f in features:
            features[f] = torch.tensor(features[f] + [0] * (max_seq_len - len(token_ids)), dtype=torch.int)
        if example.label is not None:
            features['labels'] = torch.tensor(example.label, dtype=torch.int)
        return features

    def get_model_class_fn(self):
        def partial_class(*wargs, **kwargs):
            model = GenerativeAdversarialTrainingModel.load_model(*wargs, **kwargs)
            if self.args.init_generator is not None:
                logger.info(f'Load generator from {self.args.init_generator}')
                generator = torch.load(self.args.init_generator, map_location='cpu')
                missing_keys, unexpected_keys = model.generator.load_state_dict(generator, strict=False)
                if missing_keys and (len(missing_keys) > 0):
                    logger.warning(f'Load generator with missing keys: {missing_keys}')
                if unexpected_keys and (len(unexpected_keys) > 0):
                    logger.warning(f'Load generator with unexptected keys: {unexpected_keys}')
                for name, param in model.generator.named_parameters():
                    if name not in ['deberta.embeddings.word_embeddings.weight', 'deberta.embeddings.position_embeddings.weight']:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
            if self.args.init_discriminator is not None:
                logger.info(f'Load discriminator from {self.args.init_discriminator}')
                discriminator = torch.load(self.args.init_discriminator, map_location='cpu')
                missing_keys, unexpected_keys = model.discriminator.load_state_dict(discriminator, strict=False)
                if missing_keys and (len(missing_keys) > 0):
                    logger.warning(f'Load discriminator with missing keys: {missing_keys}')
                if unexpected_keys and (len(unexpected_keys) > 0):
                    logger.warning(f'Load discriminator with unexptected keys: {unexpected_keys}')
                for param in model.discriminator.parameters():
                    param.requires_grad = True
            model.register_discriminator_fw_hook()  # share embedding
            return model

        return partial_class

    @classmethod
    def _read_json(cls, input_file, task):
        """Read AdvGLUE dev json file
          return the dev data of task
        """
        with open(input_file) as fin:
            dataset = json.load(fin)
        return dataset[task]

    @classmethod
    def add_arguments(cls, parser):
        """Add task specific arguments
          e.g. parser.add_argument('--data_dir', type=str, help='The path of data directory.')
        """
        parser.add_argument('--init_generator', type=str, default=None,
                            help='The model that used to initialize the generator')
        parser.add_argument('--init_discriminator', type=str, default=None,
                            help='The model that used to initialize the discriminator')
