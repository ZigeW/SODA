#! /usr/bin/env bash

python offline_test_time.py \
--mode da \
--method soda \
--workers 0 \
--data_root ./data/CIFAR-100 \
--dataset cifar100 \
--testdoms gaussian_noise5 \
--trained_model ./results/cifar100_joint_resnet50/joint_resnet50.pth \
--model ResNet50 \
--batch_size 256 \
--optim SGD \
--lr 0.001 \
--steps 150 \
--q 10 \
--zo 1