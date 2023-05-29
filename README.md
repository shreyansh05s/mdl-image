# SWIN Transformers for Cost-Efficient Image Classification

## Table of Contents

- [mdl-image](#mdl-image)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Installation](#installation)
  - [Training](#training)
  - [Evaluation](#evaluation)

## Overview

This is a Pytorch implementation of Huggingface models for image classification.
Models include:
- [ViT](https://arxiv.org/abs/2010.11929)
- [Swin Transformer](https://arxiv.org/abs/2103.14030)
- [ResNet](https://arxiv.org/abs/1512.03385)

Datasets included:
- [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html)

Optimizer:
- [AdamW](https://arxiv.org/abs/1711.05101)
- [SAM](https://arxiv.org/abs/2010.01412)

## Installation

First insure python 3.8.10 is installed.

Then create a virtual environment.

```bash
python -m venv venv
```

Activate the virtual environment.

```bash
source venv/bin/activate
```

Install the requirements.

```bash
pip install -r requirements.txt
```

### Additional setup for wandb logging

- If you want to use wandb logging, you will need to create an account at [wandb.ai](https://wandb.ai/).

- Then get your API key from [wandb.ai/authorize](https://wandb.ai/authorize).

- Then login to wandb from the command line.

```bash
wandb login
```

## Training

To train a model, we can call the `train.py` script, whose arguments can be found by running the following:

```bash
python train.py --help
```

For example, to train a ViT model on CIFAR100, we can run the following:

```bash
python train.py --model vit --batch-size 32 --epochs 5 --lr 0.01 --optimizer sam
```

For training with wandb logging, we can run the following:

```bash
python train.py --model vit --batch-size 32 --epochs 5 --lr 0.01 --optimizer sam -W
```


## Evaluation

To evaluate a model, we can call the `eval.py` script, whose arguments can be found by running the following:

```bash
python eval.py --help
```

For example, to evaluate a ViT model on CIFAR100, we can run the following:

```bash
python eval.py --model VIT
```

For Evaluation on all models, we can run the following:

```bash
python eval.py
```




