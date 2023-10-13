#! /usr/bin/env bash

python online_test_time.py \
--workers 0 \
--data_root ./data/CIFAR-10 \
--dataset cifar10 \
--testdoms gaussian_noise5 \
--trained_model ./results/cifar10_joint_resnet50/joint_resnet50.pth \
--model ResNet50 \
--batch_size 128 \
--optim SGD \
--lr 0.001 \
--steps 10 \
--q 5 \
--zo 1 \
--queue_size 1000