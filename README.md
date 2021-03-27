# Siamese-Neural-Networks

## Introduction
Code used for my master thesis. Comparison of two different Siamese neural networks for image recognition: 
1. Keras MNIST Siamese example and
2. [Siamese neural network for oneshot image recognition by Koch et al.](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)

## Project structure

MNIST project folder contains scripts for training and testing of Keras' siamese neural network.

OMNIGLOT project folder contains scripts for running siamese neural network by Koch et al.

## Installation

First, create conda and activate environment with:
```bash
conda create -n tf_venv python=3.7.7

conda activate tf_venv
```

After that, you can install requirements by running:

```bash
conda install -c conda-forge --file requirements.txt
```

#TODO
explain the experiment (edit readme.md)

#TODO
code cleaning?