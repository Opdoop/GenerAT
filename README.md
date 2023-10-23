# GenerativeAT

This is the source code for the paper `Generative Adversarial Training with Perturbed Token Detection for Robustness` . This project is build on [DeBERTa-V3](https://github.com/microsoft/DeBERTa) and has tested on `Ubuntu 20.04.5 LTS` with single GPU (V100 32GB).

### Prepare Environment

1. Create environment and install requirement packages using provided `environment.yml`:

```
conda env create -f environment.yml
conda activate GenerAT
```

2. Download pre-trained model
   * Download `pytorch_model.bin` and `pytorch_model.generator.bin` from [huggingface](https://huggingface.co/microsoft/deberta-v3-large/tree/main) and put it in `./deberta-v3-large` . 
3. Download glue data

```
python download_glue_data.py
```



### Train

Run the following bash scripts, it will train the model on corresponding dataset and report evaluation metrics.

* adv-rte

```
bash ./adv_glue/rte.sh
```

* adv-sst-2

```
bash ./adv_glue/sst2.sh
```

* adv-mnli

```
bash ./adv_glue/mnli.sh
```

* adv-qnli

```
bash ./adv_glue/qnli.sh
```

* adv-qqp

```
bash ./adv_glue/qqp.sh
```

