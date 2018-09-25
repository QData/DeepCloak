# DeepCloak: Masking Deep Neural Network Models for Robustness Against Adversarial Samples

## Environment: Torch7 + CUDNN

## Reference 


```
@article{GaoWQ17,
  author    = {Ji Gao and
               Beilun Wang and
               Yanjun Qi},
  title     = {DeepCloak: Masking {DNN} Models for robustness against adversarial
               samples},
  journal   = {CoRR},
  volume    = {abs/1702.06763},
  year      = {2017},
  url       = {http://arxiv.org/abs/1702.06763},
  archivePrefix = {arXiv},
  eprint    = {1702.06763},
  biburl    = {https://dblp.org/rec/bib/journals/corr/GaoWQ17},
}
```

## Example:

th removenode.lua -dataset resources/cifar10.t7 -model resources/model_res-164.t7 -layernum 8

## Usage: 

th removenode.lua -model MODELADD -dataset DATASETADD -layernum LAYERNUM -std STD [-power POWER] [-gpu GPUNUM] 

 * [MODELADD]: address of the model file \n

 * [LAYERNUM]: number of the layer where the mask will be inserted after it

 * [POWER]: attack strength, epsilon in Fast Gradient Sign Method, default 10 

 * [GPUNUM]: number of GPU selected

 * [DATASETADD]: address of the dataset file

 * [STD]: the standard deviation of the dataset used in the preprocessing, required in the Adversarial Sample Generation

## Dataset and models: Orginially from https://github.com/szagoruyko/wide-residual-networks

* Download them from http://www.cs.virginia.edu/~jg6yd/resources/

* Cifar-10: Whitened data of CIFAR-10, std = 17.067, which is selected as the default dataset

* model_vgg_orig.t7: A pretrained model of VGG-16

* model_res-164.t7: A pretrained model of the 152 layer residual network

* Wide.t7: A pretrained model of wide residual network from https://github.com/szagoruyko/wide-residual-networks
