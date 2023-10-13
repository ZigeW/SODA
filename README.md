# SODA

The implementation of SODA on CIFAR-10-C, CIFAR-100-C and ImageNet-C.

## Prerequisites:

- python == 3.10.8
- cudatoolkit == 11.7
- pytorch ==1.13.1
- torchvision == 0.14.1
- numpy, PIL, argparse, collections, math, random

## Datasets

Please download and organize [CIFAR-10-C](https://zenodo.org/record/2535967/files/CIFAR-10-C.tar?download=1), [CIFAR-100-C](https://zenodo.org/record/3555552/files/CIFAR-100-C.tar?download=1) and [ImageNet-C](https://zenodo.org/record/2235448) in this structure:

(ImageNet-C data can also be generated following instructions in [this repository](https://github.com/Gorilla-Lab-SCUT/TTAC/tree/master/imagenet))

```
BETA
├── data
    ├──CIFAR-10
    │   ├── CIFAR-10-C
    │   │   ├── brightness.npy
    │   │   ├── contrast.npy
    │   │   ├── ...
    │   │   ├── labels.npy
    ├──CIFAR-100
    │   ├── CIFAR-100-C
    │   │   ├── brightness.npy
    │   │   ├── contrast.npy
    │   │   ├── ...
    │   │   ├── labels.npy
    ├──ImageNet
    │   ├── ImageNet-C
    │   │   ├── brightness.pth
    │   │   ├── contrast.pth
    │   │   ├── ...
    │   │   ├── labels.pth
```

## Pre-trained Models

The checkpoints of pre-trained Resnet-50 can be downloaded (197MB) using the following command:

```
mkdir -p results/cifar10_joint_resnet50 && cd results/cifar10_joint_resnet50
gdown https://drive.google.com/uc?id=1MZN19o-5b2w-BI1ObIlnsJ8XBZvMuL77 && cd ../..
mkdir -p results/cifar100_joint_resnet50 && cd results/cifar100_joint_resnet50
gdown https://drive.google.com/uc?id=1C7knE2S9kKDYZrqd4Bo4S5lOgp7Le_DP && cd ../..
mkdir -p results/imagenet && cd results/imagenet
gdown https://drive.google.com/uc?id=1GSGzOv0MNBBMEYeRRQlp1WGD1USDl0iP && cd ../..
```

The CIFAR-10/100 pre-trained models are obtained by training on the clean CIFAR-10/100 images using [semi-supervised SimCLR](https://github.com/YuejiangLIU/semi-simclr). The ImageNet pre-trained model is obtained from [TorchVision](https://pytorch.org/vision/stable/models.html)

## Adaptation on CIFAR-10-C

```
# offline SODA
bash scripts/run_offline_soda_10.sh

# offline SODA-R
bash scripts/run_offline_soda_r_10.sh

# offline MA-SO
bash scripts/run_offline_ma_10.sh

# online SODA-O
bash scripts/run_online_soda_10.sh
```

## Adaptation on CIFAR-100-C

```
# offline SODA
bash scripts/run_offline_soda_100.sh

# offline SODA-R
bash scripts/run_offline_soda_r_100.sh

# offline MA-SO
bash scripts/run_offline_ma_100.sh

# online SODA-O
bash scripts/run_online_soda_100.sh
```

## Adaptation on ImageNet-C

```
# offline SODA
bash scripts/run_offline_soda_imagenet.sh

# offline SODA-R
bash scripts/run_offline_soda_r_imagenet.sh

# offline MA-SO
bash scripts/run_offline_ma_imagenet.sh
```

