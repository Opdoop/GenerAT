#     
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Author:   
# Date:
#

import os
from collections import OrderedDict

from .metrics import *
from .task import EvalData, Task
from .mlm_task import NGramMaskGenerator
from .task_registry import register_task
from ...data import DynamicDataset
from ...data.example import *
from ...utils import get_logger


logger = get_logger()

__all__ = ["AdvMNLITask", "AdvSST2Task", "AdvQQPTask", "AdvRTETask", "AdvQNLITask"]


@register_task("adv-rte")
class AdvRTETask(Task):
    def __init__(self, data_dir, tokenizer, args, **kwargs):
        super().__init__(tokenizer, args, **kwargs)
        self.mask_gen = NGramMaskGenerator(tokenizer, max_gram=1, keep_prob=0, mask_prob=1,
                                           max_seq_len=args.max_seq_length)
        self.data_dir = data_dir

    def train_data(self, max_seq_len=512, dataset_size=None, epochs=1, mask_gen=None, **kwargs):
        input_src = os.path.join(self.data_dir, 'train.tsv')
        assert os.path.exists(input_src), f"{input_src} doesn't exists"
        data = self._read_tsv(input_src)
        examples = [ExampleInstance((l[1], l[2]), self.label2id(l[3])) for l in data[1:]]  # if l[3] in ['slate']])

        examples = ExampleSet(examples)
        if dataset_size is None:
            dataset_size = len(examples) * epochs
        return DynamicDataset(examples, feature_fn=self.get_feature_fn(max_seq_len=max_seq_len, mask_gen=self.mask_gen), \
                              dataset_size=dataset_size, shuffle=True, **kwargs)


    def eval_data(self, max_seq_len=512, dataset_size=None, extra_data=None, **kwargs):
        ds = [
            self._data('dev', "dev.tsv", 'dev'),
        ]

        if extra_data is not None:
            extra_data = extra_data.split(',')
            for d in extra_data:
                n, path = d.split(':')
                ds.append(self._data(n, path, 'dev+'))

        for d in ds:
            if dataset_size is None:
                _size = len(d.data)
            else:
                _size = dataset_size
            d.data = DynamicDataset(d.data, feature_fn=self.get_feature_fn(max_seq_len=max_seq_len), dataset_size=_size,
                                    **kwargs)
        return ds

    def test_data(self, max_seq_len=512, dataset_size=None, **kwargs):
        """See base class."""
        ds = [
            self._data('test', 'test.tsv', 'test')
        ]
        for d in ds:
            if dataset_size is None:
                _size = len(d.data)
            else:
                _size = dataset_size
            d.data = DynamicDataset(d.data, feature_fn=self.get_feature_fn(max_seq_len=max_seq_len), dataset_size=_size,
                                    **kwargs)
        return ds

    def adv_eval_data(self, name, path, max_seq_len=512, dataset_size=None, **kwargs):
        assert os.path.exists(path), f"{path} doesn't exists"
        data = self._read_json(path, 'rte')
        examples = ExampleSet([ExampleInstance((l['sentence1'], l['sentence2']), int(l['label'])) for l in data])
        predict_fn = self.get_predict_fn()

        ds = [EvalData(name, examples,
                       metrics_fn=self.get_metrics_fn(), predict_fn=predict_fn)]
        for d in ds:
            if dataset_size is None:
                _size = len(d.data)
            else:
                _size = dataset_size
            d.data = DynamicDataset(d.data, feature_fn=self.get_feature_fn(max_seq_len=max_seq_len, mask_gen=self.mask_gen), dataset_size=_size,
                                    **kwargs)
        return ds

    def adv_test_data(self, path, max_seq_len=512, dataset_size=None, **kwargs):
        assert os.path.exists(path), f"{path} doesn't exists"
        data = self._read_json(path, 'rte')
        examples = ExampleSet([ExampleInstance((l['sentence1'], l['sentence2'])) for l in data])
        predict_fn = self.get_adv_predict_fn()

        ds = [EvalData('rte', examples,
                       metrics_fn=self.get_metrics_fn(), predict_fn=predict_fn)]
        for d in ds:
            if dataset_size is None:
                _size = len(d.data)
            else:
                _size = dataset_size
            d.data = DynamicDataset(d.data, feature_fn=self.get_feature_fn(max_seq_len=max_seq_len), dataset_size=_size,
                                    **kwargs)
        return ds

    def _data(self, name, path, type_name='dev'):
        input_src = os.path.join(self.data_dir, path)
        assert os.path.exists(input_src), f"{input_src} doesn't exists"
        data = self._read_tsv(input_src)
        if type_name == 'test':
            examples = ExampleSet([ExampleInstance((l[1], l[2])) for l in data[1:]])
        else:
            examples = ExampleSet([ExampleInstance((l[1], l[2]), self.label2id(l[3])) for l in data[1:]])

        predict_fn = self.get_predict_fn()
        return EvalData(name, examples,
                        metrics_fn=self.get_metrics_fn(), predict_fn=predict_fn)

    def get_metrics_fn(self):
        """Calcuate metrics based on prediction results"""

        def metrics_fn(logits, labels):
            return OrderedDict(accuracy=metric_accuracy(logits, labels))

        return metrics_fn

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]


@register_task('adv-qnli')
class AdvQNLITask(Task):
    def __init__(self, data_dir, tokenizer, args, **kwargs):
        super().__init__(tokenizer, args, **kwargs)
        self.data_dir = data_dir
        self.mask_gen = NGramMaskGenerator(tokenizer, max_gram=1, keep_prob=0, mask_prob=1,
                                           max_seq_len=args.max_seq_length)

    def train_data(self, max_seq_len=512, dataset_size=None, epochs=1, mask_gen=None, **kwargs):
        input_src = os.path.join(self.data_dir, 'train.tsv')
        assert os.path.exists(input_src), f"{input_src} doesn't exists"
        data = self._read_tsv(input_src)
        examples = ExampleSet(
            [ExampleInstance((l[2], l[1]), self.label2id(l[3])) for l in data[1:]])  # if l[3] in ['slate']])
        if dataset_size is None:
            dataset_size = len(examples) * epochs
        return DynamicDataset(examples, feature_fn=self.get_feature_fn(max_seq_len=max_seq_len, mask_gen=self.mask_gen), \
                              dataset_size=dataset_size, shuffle=True, **kwargs)

    def eval_data(self, max_seq_len=512, dataset_size=None, **kwargs):
        ds = [
            self._data('dev', "dev.tsv", 'dev')
        ]
        for d in ds:
            if dataset_size is None:
                _size = len(d.data)
            else:
                _size = dataset_size
            d.data = DynamicDataset(d.data, feature_fn=self.get_feature_fn(max_seq_len=max_seq_len), dataset_size=_size,
                                    **kwargs)
        return ds

    def test_data(self, max_seq_len=512, dataset_size=None, **kwargs):
        """See base class."""
        ds = [
            self._data('test', 'test.tsv', 'test')
        ]
        for d in ds:
            if dataset_size is None:
                _size = len(d.data)
            else:
                _size = dataset_size
            d.data = DynamicDataset(d.data, feature_fn=self.get_feature_fn(max_seq_len=max_seq_len), dataset_size=_size,
                                    **kwargs)
        return ds

    def adv_eval_data(self, name, path, max_seq_len=512, dataset_size=None, **kwargs):
        assert os.path.exists(path), f"{path} doesn't exists"
        data = self._read_json(path, 'qnli')
        examples = ExampleSet([ExampleInstance((l['sentence'], l['question']), int(l['label'])) for l in data])
        predict_fn = self.get_predict_fn()  # 保存 submit 文件的函数

        ds = [EvalData(name, examples,
                       metrics_fn=self.get_metrics_fn(), predict_fn=predict_fn)]
        for d in ds:
            if dataset_size is None:
                _size = len(d.data)
            else:
                _size = dataset_size
            d.data = DynamicDataset(d.data, feature_fn=self.get_feature_fn(max_seq_len=max_seq_len), dataset_size=_size,
                                    **kwargs)
        return ds

    def adv_test_data(self, path, max_seq_len=512, dataset_size=None, **kwargs):
        assert os.path.exists(path), f"{path} doesn't exists"
        data = self._read_json(path, 'qnli')
        examples = ExampleSet([ExampleInstance((l['sentence'], l['question'])) for l in data])
        predict_fn = self.get_adv_predict_fn()  # 保存 submit 文件的函数

        ds = [EvalData('qnli', examples,
                       metrics_fn=self.get_metrics_fn(), predict_fn=predict_fn)]
        for d in ds:
            if dataset_size is None:
                _size = len(d.data)
            else:
                _size = dataset_size
            d.data = DynamicDataset(d.data, feature_fn=self.get_feature_fn(max_seq_len=max_seq_len), dataset_size=_size,
                                    **kwargs)
        return ds

    def _data(self, name, path, type_name='dev'):
        input_src = os.path.join(self.data_dir, path)
        assert os.path.exists(input_src), f"{input_src} doesn't exists"
        data = self._read_tsv(input_src)
        predict_fn = self.get_predict_fn()
        if type_name == 'test':
            examples = ExampleSet([ExampleInstance((l[2], l[1])) for l in data[1:]])
        else:
            examples = ExampleSet([ExampleInstance((l[2], l[1]), self.label2id(l[3])) for l in data[1:]])

        return EvalData(name, examples,
                        metrics_fn=self.get_metrics_fn(), predict_fn=predict_fn)

    def get_metrics_fn(self):
        """Calcuate metrics based on prediction results"""

        def metrics_fn(logits, labels):
            return OrderedDict(accuracy=metric_accuracy(logits, labels))

        return metrics_fn

    def get_predict_fn(self):
        """Calcuate metrics based on prediction results"""

        def predict_fn(logits, output_dir, name, prefix):
            output = os.path.join(output_dir, 'submit-{}-{}.tsv'.format(name, prefix))
            preds = np.argmax(logits, axis=1)
            labels = self.get_labels()
            with open(output, 'w', encoding='utf-8') as fs:
                fs.write('index\tpredictions\n')
                for i, p in enumerate(preds):
                    fs.write('{}\t{}\n'.format(i, labels[p]))

        return predict_fn

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]


@register_task('adv-sst-2')
class AdvSST2Task(Task):
    def __init__(self, data_dir, tokenizer, args, **kwargs):
        super().__init__(tokenizer, args, **kwargs)
        self.data_dir = data_dir
        self.mask_gen = NGramMaskGenerator(tokenizer, max_gram=1, keep_prob=0, mask_prob=1,
                                           max_seq_len=args.max_seq_length)

    def train_data(self, max_seq_len=512, dataset_size=None, epochs=1, mask_gen=None, **kwargs):
        input_src = os.path.join(self.data_dir, 'train.tsv')
        assert os.path.exists(input_src), f"{input_src} doesn't exists"
        data = self._read_tsv(input_src)
        examples = ExampleSet(
            [ExampleInstance((l[0],), self.label2id(l[1])) for l in data[1:]])  # if l[3] in ['slate']])
        # data[1:]，第一行是 title
        # l[0] 是 sentence, l[1] 是 label
        if dataset_size is None:
            dataset_size = len(examples) * epochs
        # 封装为 DynamicDataset, get_feature_fn 为 tokenization
        return DynamicDataset(examples, feature_fn=self.get_feature_fn(max_seq_len=max_seq_len, mask_gen=self.mask_gen), \
                              dataset_size=dataset_size, shuffle=True, **kwargs)

    def eval_data(self, max_seq_len=512, dataset_size=None, **kwargs):
        ds = [
            self._data('dev', 'dev.tsv', 'dev')
        ]
        for d in ds:
            if dataset_size is None:
                _size = len(d.data)
            else:
                _size = dataset_size
            d.data = DynamicDataset(d.data, feature_fn=self.get_feature_fn(max_seq_len=max_seq_len), dataset_size=_size,
                                    **kwargs)
        return ds

    def test_data(self, max_seq_len=512, dataset_size=None, **kwargs):
        """See base class."""
        ds = [
            self._data('test', 'test.tsv', 'test')  # 调用的下面的 _data 函数
        ]
        for d in ds:
            if dataset_size is None:
                _size = len(d.data)
            else:
                _size = dataset_size
            d.data = DynamicDataset(d.data, feature_fn=self.get_feature_fn(max_seq_len=max_seq_len), dataset_size=_size,
                                    **kwargs)
        return ds

    def adv_eval_data(self, name, path, max_seq_len=512, dataset_size=None, **kwargs):
        assert os.path.exists(path), f"{path} doesn't exists"
        data = self._read_json(path, 'sst2')
        examples = ExampleSet([ExampleInstance((l['sentence'],), int(l['label'])) for l in data])
        predict_fn = self.get_predict_fn()  # 保存 submit 文件的函数

        ds = [EvalData(name, examples,
                       metrics_fn=self.get_metrics_fn(), predict_fn=predict_fn)]
        for d in ds:
            if dataset_size is None:
                _size = len(d.data)
            else:
                _size = dataset_size
            d.data = DynamicDataset(d.data, feature_fn=self.get_feature_fn(max_seq_len=max_seq_len), dataset_size=_size,
                                    **kwargs)
        return ds

    def adv_test_data(self, path, max_seq_len=512, dataset_size=None, **kwargs):
        assert os.path.exists(path), f"{path} doesn't exists"
        data = self._read_json(path, 'sst2')
        examples = ExampleSet([ExampleInstance((l['sentence'],)) for l in data])
        predict_fn = self.get_adv_predict_fn()  # 保存 submit 文件的函数

        ds = [EvalData('sst2', examples,
                       metrics_fn=self.get_metrics_fn(), predict_fn=predict_fn)]
        for d in ds:
            if dataset_size is None:
                _size = len(d.data)
            else:
                _size = dataset_size
            d.data = DynamicDataset(d.data, feature_fn=self.get_feature_fn(max_seq_len=max_seq_len), dataset_size=_size,
                                    **kwargs)
        return ds

    def _data(self, name, path, type_name='dev'):
        input_src = os.path.join(self.data_dir, path)
        assert os.path.exists(input_src), f"{input_src} doesn't exists"
        data = self._read_tsv(input_src)
        predict_fn = self.get_predict_fn()  # 保存 submit 文件的函数
        if type_name == 'test':  # test 没有 label
            examples = ExampleSet([ExampleInstance((l[1],)) for l in data[1:]])
        elif type_name == 'orig-test':
            examples = ExampleSet([ExampleInstance((l[1],), self.label2id(l[3])) for l in data[1:]])
        else:  # dev
            examples = ExampleSet([ExampleInstance((l[0],), self.label2id(l[1])) for l in data[1:]])

        # 封装成 EvalData，扩展了 metrics_fn 和 predict_fn
        return EvalData(name, examples,
                        metrics_fn=self.get_metrics_fn(), predict_fn=predict_fn)

    def get_metrics_fn(self):
        """Calcuate metrics based on prediction results"""

        def metrics_fn(logits, labels):
            return OrderedDict(accuracy=metric_accuracy(logits, labels))

        return metrics_fn

    def get_labels(self):
        """See base class."""
        return ["0", "1"]


@register_task('adv-qqp')
class AdvQQPTask(Task):
    def __init__(self, data_dir, tokenizer, args, **kwargs):
        super().__init__(tokenizer, args, **kwargs)
        self.data_dir = data_dir
        self.mask_gen = NGramMaskGenerator(tokenizer, max_gram=1, keep_prob=0, mask_prob=1,
                                           max_seq_len=args.max_seq_length)

    def train_data(self, max_seq_len=512, dataset_size=None, epochs=1, mask_gen=None, **kwargs):
        input_src = os.path.join(self.data_dir, 'train.tsv')
        assert os.path.exists(input_src), f"{input_src} doesn't exists"
        data = self._read_tsv(input_src)
        examples = ExampleSet([ExampleInstance((l[3], l[4]), self.label2id(l[5])) for l in data[1:] if
                               len(l) == 6])  # if l[3] in ['slate']])
        if dataset_size is None:
            dataset_size = len(examples) * epochs
        return DynamicDataset(examples, feature_fn=self.get_feature_fn(max_seq_len=max_seq_len, mask_gen=self.mask_gen), \
                              dataset_size=dataset_size, shuffle=True, **kwargs)

    def eval_data(self, max_seq_len=512, dataset_size=None, **kwargs):
        ds = [
            self._data('dev', 'dev.tsv', 'dev')
        ]
        for d in ds:
            if dataset_size is None:
                _size = len(d.data)
            else:
                _size = dataset_size
            d.data = DynamicDataset(d.data, feature_fn=self.get_feature_fn(max_seq_len=max_seq_len), dataset_size=_size,
                                    **kwargs)
        return ds

    def test_data(self, max_seq_len=512, dataset_size=None, **kwargs):
        """See base class."""
        ds = [
            self._data('test', 'test.tsv', 'test')
        ]
        for d in ds:
            if dataset_size is None:
                _size = len(d.data)
            else:
                _size = dataset_size
            d.data = DynamicDataset(d.data, feature_fn=self.get_feature_fn(max_seq_len=max_seq_len), dataset_size=_size,
                                    **kwargs)
        return ds

    def adv_eval_data(self, name, path, max_seq_len=512, dataset_size=None, **kwargs):
        assert os.path.exists(path), f"{path} doesn't exists"
        data = self._read_json(path, 'qqp')
        examples = ExampleSet([ExampleInstance((l['question1'], l['question2']), int(l['label'])) for l in data])
        predict_fn = self.get_predict_fn()  # 保存 submit 文件的函数

        ds = [EvalData(name, examples,
                       metrics_fn=self.get_metrics_fn(), predict_fn=predict_fn)]
        for d in ds:
            if dataset_size is None:
                _size = len(d.data)
            else:
                _size = dataset_size
            d.data = DynamicDataset(d.data, feature_fn=self.get_feature_fn(max_seq_len=max_seq_len), dataset_size=_size,
                                    **kwargs)
        return ds

    def adv_test_data(self, path, max_seq_len=512, dataset_size=None, **kwargs):
        assert os.path.exists(path), f"{path} doesn't exists"
        data = self._read_json(path, 'qqp')
        examples = ExampleSet([ExampleInstance((l['question1'], l['question2'])) for l in data])
        predict_fn = self.get_adv_predict_fn()  # 保存 submit 文件的函数

        ds = [EvalData('qqp', examples,
                       metrics_fn=self.get_metrics_fn(), predict_fn=predict_fn)]
        for d in ds:
            if dataset_size is None:
                _size = len(d.data)
            else:
                _size = dataset_size
            d.data = DynamicDataset(d.data, feature_fn=self.get_feature_fn(max_seq_len=max_seq_len), dataset_size=_size,
                                    **kwargs)
        return ds

    def _data(self, name, path, type_name='dev'):
        input_src = os.path.join(self.data_dir, path)
        assert os.path.exists(input_src), f"{input_src} doesn't exists"
        data = self._read_tsv(input_src)
        predict_fn = self.get_predict_fn()
        if type_name == 'test':
            examples = ExampleSet([ExampleInstance((l[-2], l[-1])) for l in data[1:]])
        else:
            examples = ExampleSet([ExampleInstance((l[3], l[4]), self.label2id(l[5])) for l in data[1:] if len(l) == 6])

        return EvalData(name, examples,
                        metrics_fn=self.get_metrics_fn(), predict_fn=predict_fn)

    def get_metrics_fn(self):
        """Calcuate metrics based on prediction results"""

        def metrics_fn(logits, labels):
            return OrderedDict(accuracy=metric_accuracy(logits, labels),
                               f1=metric_f1(logits, labels))

        return metrics_fn

    def get_labels(self):
        """See base class."""
        return ["0", "1"]


@register_task('adv-mnli')
class AdvMNLITask(Task):
    def __init__(self, data_dir, tokenizer, args, **kwargs):
        super().__init__(tokenizer, args, **kwargs)
        self.data_dir = data_dir
        self.mask_gen = NGramMaskGenerator(tokenizer, max_gram=1, keep_prob=0, mask_prob=1,
                                           max_seq_len=args.max_seq_length)

    def train_data(self, max_seq_len=512, dataset_size=None, epochs=1, mask_gen=None, **kwargs):
        input_src = os.path.join(self.data_dir, 'train.tsv')
        assert os.path.exists(input_src), f"{input_src} doesn't exists"
        data = self._read_tsv(input_src)
        examples = [ExampleInstance((l[8], l[9]), self.label2id(l[-1])) for l in data[1:]]  # if l[3] in ['slate']])
        examples = ExampleSet(examples)
        if dataset_size is None:
            dataset_size = len(examples) * epochs
        return DynamicDataset(examples, feature_fn=self.get_feature_fn(max_seq_len=max_seq_len, mask_gen=self.mask_gen), \
                              dataset_size=dataset_size, shuffle=True, **kwargs)

    def eval_data(self, max_seq_len=512, dataset_size=None, **kwargs):
        ds = [
            self._data('matched', 'dev_matched.tsv', 'dev'),
            self._data('mismatched', 'dev_mismatched.tsv', 'dev'),
        ]

        for d in ds:
            if dataset_size is None:
                _size = len(d.data)
            d.data = DynamicDataset(d.data, feature_fn=self.get_feature_fn(max_seq_len=max_seq_len), dataset_size=_size,
                                    **kwargs)
        return ds

    def test_data(self, max_seq_len=512, dataset_size=None, **kwargs):
        """See base class."""
        ds = [
            self._data('matched', 'test_matched.tsv', 'test'),
            self._data('mismatched', 'test_mismatched.tsv', 'test'),
        ]

        for d in ds:
            if dataset_size is None:
                _size = len(d.data)
            d.data = DynamicDataset(d.data, feature_fn=self.get_feature_fn(max_seq_len=max_seq_len), dataset_size=_size,
                                    **kwargs)
        return ds

    def diagnostic_data(self, name, path, type_name='dev', ignore_metric=False):
        input_src = os.path.join(self.data_dir, path)
        assert os.path.exists(input_src), f"{input_src} doesn't exists"
        data = self._read_tsv(input_src)
        predict_fn = self.get_predict_fn()
        examples = ExampleSet([ExampleInstance((l[5], l[6]), self.label2id(l[7])) for l in data[1:]])

        def _metric_fn(logits, labels):
            return OrderedDict(
                accuracy=metric_accuracy(logits, labels),
                mcc=metric_mcc(logits, labels))

        return EvalData(name, examples,
                        metrics_fn=_metric_fn, predict_fn=predict_fn, ignore_metric=ignore_metric,
                        critial_metrics=['mcc'])

    def adv_eval_data(self, name, path, max_seq_len=512, dataset_size=None, ignore_metric=False, **kwargs):
        assert os.path.exists(path), f"{path} doesn't exists"
        matched_data = self._read_json(path, 'mnli')
        matched_examples = ExampleSet(
            [ExampleInstance((l['premise'], l['hypothesis']), int(l['label'])) for l in matched_data])

        mismatched_data = self._read_json(path, 'mnli-mm')
        mismatched_examples = ExampleSet(
            [ExampleInstance((l['premise'], l['hypothesis']), int(l['label'])) for l in mismatched_data])

        predict_fn = self.get_predict_fn()  # 保存 submit 文件的函数

        ds = [EvalData(name + 'matched', matched_examples,
                       predict_fn=predict_fn, ignore_metric=ignore_metric, critial_metrics=['accuracy']),
              EvalData(name + 'mismatched', mismatched_examples,
                       predict_fn=predict_fn, ignore_metric=ignore_metric, critial_metrics=['accuracy'])
              ]
        for d in ds:
            if dataset_size is None:
                _size = len(d.data)
            else:
                _size = dataset_size
            d.data = DynamicDataset(d.data, feature_fn=self.get_feature_fn(max_seq_len=max_seq_len), dataset_size=_size,
                                    **kwargs)
        return ds

    def adv_test_data(self, path, max_seq_len=512, dataset_size=None, ignore_metric=False, **kwargs):
        assert os.path.exists(path), f"{path} doesn't exists"
        matched_data = self._read_json(path, 'mnli')
        matched_examples = ExampleSet([ExampleInstance((l['premise'], l['hypothesis'])) for l in matched_data])

        mismatched_data = self._read_json(path, 'mnli-mm')
        mismatched_examples = ExampleSet([ExampleInstance((l['premise'], l['hypothesis'])) for l in mismatched_data])

        predict_fn = self.get_adv_predict_fn()  # 保存 submit 文件的函数

        ds = [EvalData('mnli', matched_examples,
                       predict_fn=predict_fn, ignore_metric=ignore_metric, critial_metrics=['accuracy']),
              EvalData('mnli-mm', mismatched_examples,
                       predict_fn=predict_fn, ignore_metric=ignore_metric, critial_metrics=['accuracy'])
              ]
        for d in ds:
            if dataset_size is None:
                _size = len(d.data)
            else:
                _size = dataset_size
            d.data = DynamicDataset(d.data, feature_fn=self.get_feature_fn(max_seq_len=max_seq_len), dataset_size=_size,
                                    **kwargs)
        return ds

    def anli_data(self, name, path, type_name='dev', ignore_metric=False):
        input_src = os.path.join(self.data_dir, path)
        assert os.path.exists(input_src), f"{input_src} doesn't exists"
        data = self._read_tsv(input_src)
        predict_fn = self.get_predict_fn()
        examples = ExampleSet([ExampleInstance((l[1], l[2]), self.label2id(l[3])) for l in data[1:]])

        def _metric_fn(logits, labels):
            return OrderedDict(
                accuracy=metric_accuracy(logits, labels))

        return EvalData(name, examples,
                        metrics_fn=_metric_fn, predict_fn=predict_fn, ignore_metric=ignore_metric,
                        critial_metrics=['accuracy'])

    def _data(self, name, path, type_name='dev', ignore_metric=False):
        input_src = os.path.join(self.data_dir, path)
        assert os.path.exists(input_src), f"{input_src} doesn't exists"
        data = self._read_tsv(input_src)
        predict_fn = self.get_predict_fn()
        if type_name == 'test':
            examples = ExampleSet([ExampleInstance((l[8], l[9])) for l in data[1:]])
        else:
            examples = ExampleSet([ExampleInstance((l[8], l[9]), self.label2id(l[-1])) for l in data[1:]])

        return EvalData(name, examples,
                        metrics_fn=self.get_metrics_fn(), predict_fn=predict_fn, ignore_metric=ignore_metric,
                        critial_metrics=['accuracy'])

    def get_metrics_fn(self):
        """Calcuate metrics based on prediction results"""

        def metrics_fn(logits, labels):
            metrics = OrderedDict(accuracy=metric_accuracy(logits, labels))
            metrics[f'accuracy_all'] = metric_accuracy(logits, labels)
            return metrics

        return metrics_fn

    def get_labels(self):
        """See base class."""
        return ["entailment", "neutral", "contradiction"]
