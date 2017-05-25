# DeepCloak: Masking Deep Neural Network Models for Robustness Against Adversarial Samples

## Environment: Torch7 + CUDNN

## Usage: 

th removenode.lua -model MODELADD -dataset DATASETADD -layernum LAYERNUM -std STD [-power POWER] [-gpu GPUNUM] 

MODELADD: address of the model file \n

LAYERNUM: number of the layer where the mask will be inserted after it

POWER: attack strength, epsilon in Fast Gradient Sign Method, default 10 

GPUNUM: number of GPU selected

DATASETADD: address of the dataset file

STD: the standard deviation of the dataset used in the preprocessing, required in the Adversarial Sample Generation

## Dataset: Orginially from https://github.com/szagoruyko/wide-residual-networks
