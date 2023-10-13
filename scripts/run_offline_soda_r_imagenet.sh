#! /usr/bin/env bash

python offline_test_time.py \
--mode da \
--method soda \
--workers 0 1 \
--dataset imagenet \
--data_root ./data/ImageNet \
--testdoms gaussian_noise5 \
--trained_model ./results/imagenet/torchvision_resnet50.pth \
--batch_size 256 \
--optim Adam \
--lr 0.001 \
--steps 150 \
--n_resblocks 1 \
--n_downsample 1 \
--use_dropout 1 \
--zo 0 \
--tau 0.1 \
--activation relu