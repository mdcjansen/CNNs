# CNNs

This repo contains various scripts produced during my bachelors graduation internship. These include the scripts for
six CNNs to train and test, image preprocessing, and post-training analysis. 

The CNNs can be trained to perform binary classification tasks on a set of input images

## Table of contents
* [Requirements](#requirements)
* [CNN testing](#cnn-testing)
* [CNN training](#cnn-training)
* [WIP](#WIP)
* [Post training analysis](#post-training-analysis)
* [Preprocessing](#preprocessing)
* [Acknowledgements](#Acknowledgements)


## Requirements
All models were run and trained on an in-house anaconda environment. This environment mirrors the freely available
[pytorch2 environment](https://pytorch.org/get-started/pytorch-2.0/#getting-started). Additionally, 
[wandb.ai](https://wandb.ai/site) is required for logging the training and validation metrics produced by the CNNs

## CNN testing
The CNN scripts within this folder are capable of loading the models produced during training and running them on a test
set or functional dataset in order to perform predictions with the trained models.

## CNN training
Here, the six CNNs [AlexNet](https://doi.org/10.1145/3065386), [DenseNet121](https://doi.org/10.48550/arXiv.1608.06993),
[ConvNeXt Tiny](https://doi.org/10.48550/arXiv.2201.03545), [InceptionV3](https://doi.org/10.48550/arXiv.1512.00567), 
[ResNet50](https://doi.org/10.48550/arXiv.1512.03385), and [ShuffleNet](https://doi.org/10.48550/arXiv.1807.11164) 
can be found. All scripts are in working order and capable of performing binary classification tasks. These CNNs require
a csv file containing the file path and hyperparameters for the model. An example file can be found within this folder.

## Post training analysis
This folder contains all scripts that can be utilised after training or testing of the models has been completed

## Preprocessing
In this folder, various scripts can be found that aid in the preprocessing of input data, before training or testing the
CNNs.


## Acknowledgements
* [AlexNet](https://doi.org/10.1145/3065386)
* [DenseNet121](https://doi.org/10.48550/arXiv.1608.06993)
* [ConvNeXt Tiny](https://doi.org/10.48550/arXiv.2201.03545)
* [InceptionV3](https://doi.org/10.48550/arXiv.1512.00567)
* [PyTorch](https://dl.acm.org/doi/10.5555/3454287.3455008)
* [ResNet50](https://doi.org/10.48550/arXiv.1512.03385)
* [ShuffleNet](https://doi.org/10.48550/arXiv.1807.11164)
* [wandb.ai](https://wandb.ai/site)
