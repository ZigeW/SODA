#! /usr/bin/env bash

python offline_test_time.py \
--mode ma \
--method soda \
--workers 0 \
--dataset imagenet \
--data_root ./data/ImageNet \
--testdoms gaussian_noise5 \
--trained_model ./results/imagenet/torchvision_resnet50.pth \
--batch_size 256 \
--optim SGD \
--lr 0.001 \
--steps 150 \
--tau 0.1 \