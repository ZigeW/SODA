import os
import torch
from torch.utils import data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import sys
sys.path.append('..')

CIFAR10_NORM = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
CIFAR100_NORM = ((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                 (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
IMAGENET_NORM = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

CORRUPTION = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'speckle_noise',
              'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'gaussian_blur',
              'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform',
              'pixelate', 'jpeg_compression', 'spatter', 'saturate']


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class Clean_subset(data.Subset):
    def __getitem__(self, idx):
        img, _, index = self.dataset[self.indices[idx]]
        target = self.targets[idx]
        return img, target, index


class CIFAR10_idxed(datasets.CIFAR10):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        if not torch.is_tensor(img):
            img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


class CIFAR100_idxed(datasets.CIFAR100):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


class IMAGENET_idxed(datasets.ImageNet):
    def __getitem__(self, index):
        img = self.data[index, :, :, :]
        target = self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target, index


def get_data_transforms(dataset):
    if dataset.startswith('cifar'):
        if dataset == 'cifar10':
            NORM = CIFAR10_NORM
        elif dataset == 'cifar100':
            NORM = CIFAR100_NORM
        RESIZE = (32, (0.2, 1.))
    else:
        NORM = IMAGENET_NORM
        RESIZE = (224,)
    train_transforms = transforms.Compose([# transforms.RandomCrop(32, padding=4),
                                           transforms.RandomResizedCrop(*RESIZE),
                                           transforms.RandomHorizontalFlip(),
                                           # RandAugmentMC(n=2, m=10),
                                           transforms.ToTensor(),
                                           transforms.Normalize(*NORM)])

    test_transforms = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(*NORM)])

    moco_transforms = transforms.Compose([transforms.RandomResizedCrop(*RESIZE),
                                          transforms.RandomApply(
                                              [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8  # not strengthened
                                          ),
                                          transforms.RandomGrayscale(p=0.2),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(*NORM)])
    return train_transforms, test_transforms, moco_transforms


def load_corruption(data_dir, c, l, dataset):
    if dataset == 'cifar10':
        loc_dir = os.path.join(data_dir, 'CIFAR-10-C')
    elif dataset == 'cifar100':
        loc_dir = os.path.join(data_dir, 'CIFAR-100-C')
    elif dataset == 'imagenet':
        loc_dir = os.path.join(data_dir, 'ImageNet-C')

    if c not in CORRUPTION:
        raise RuntimeError("unknown corruption type")
    if dataset.startswith('cifar'):
        c_data = np.load(os.path.join(loc_dir, c + '.npy'))
        if l > 5 or l <= 0:
            raise RuntimeError("severity level out of range")
        # data = c_data[(l-1)*10000:l*10000].transpose((0, 3, 1, 2))  # total 50000 images, 0-10000 are severity level 1, ...
        data = c_data[(l - 1) * 10000:l * 10000]

        label = np.load(os.path.join(loc_dir, 'labels.npy'))
        label = label[(l - 1) * 10000:l * 10000]
    elif dataset.startswith('imagenet'):
        data = torch.load(os.path.join(loc_dir, c + '.pth')).numpy()
        label = torch.load(os.path.join(loc_dir, 'labels.pth')).numpy()
    return data, label


def load_mixtest(data_dir, level, dataset_name):
    for i, c in enumerate(CORRUPTION):
        data, label = load_corruption(data_dir, c, level, dataset_name)
        size = int(np.shape(data)[0] * 0.1)
        sample = np.random.randint(np.shape(data)[0], size=size)
        if i == 0:
            dataset = data[sample]
            labelset = label[sample]
        else:
            dataset = np.vstack((dataset, data[sample]))
            labelset = np.hstack((labelset, label[sample]))
    return dataset, labelset


def load_original(data_dir, n_bat, dataset, train_val_split=1, train=False, workers=0, shuffle=True, drop_last=False):
    train_transforms, test_transforms, _ = get_data_transforms(dataset)
    if not train:
        if dataset == 'cifar10':
            set = CIFAR10_idxed(data_dir, train=False, transform=test_transforms)
        elif dataset == 'cifar100':
            set = CIFAR100_idxed(data_dir, train=False, transform=test_transforms)
        elif dataset == 'imagenet':
            set = IMAGENET_idxed(data_dir, split='val', transform=test_transforms)
    else:
        if dataset == 'cifar10':
            set = CIFAR10_idxed(data_dir, train=True, transform=train_transforms)
        elif dataset == 'cifar100':
            set = CIFAR100_idxed(data_dir, train=True, transform=train_transforms)
        elif dataset == 'imagenet':
            set = IMAGENET_idxed(data_dir, split='train', transform=test_transforms)

    if not train:
        data_loader = data.DataLoader(set, batch_size=n_bat, shuffle=shuffle,
                                      num_workers=workers, pin_memory=True, drop_last=drop_last)
        return data_loader, None
    else:
        index = torch.randperm(len(set))
        len_train = int(len(set) * train_val_split)
        train_index, val_index = index[:len_train], index[len_train:]
        train_dataset = data.Subset(set, train_index)
        val_dataset = data.Subset(set, val_index)

        train_loader = data.DataLoader(train_dataset, batch_size=n_bat, shuffle=shuffle,
                                       num_workers=workers, pin_memory=True, drop_last=drop_last)
        val_loader = data.DataLoader(val_dataset, batch_size=n_bat, shuffle=shuffle,
                                     num_workers=workers, pin_memory=True, drop_last=drop_last)
        return train_loader, val_loader


def load_test(data_dir, n_bat, dataset, corruption=None, level=None, train_val_split=1, exclude_indices=None,
              workers=0, shuffle=True, drop_last=False, pin_memory=True, moco=False):
    if corruption == 'original':
        return load_original(data_dir, n_bat, dataset, train_val_split, shuffle, drop_last)[0]
    if corruption == 'mix':
        test_data, test_label = load_mixtest(data_dir, level, dataset)
    elif corruption in CORRUPTION:
        test_data, test_label = load_corruption(data_dir, corruption, level, dataset)
    _, test_transforms, moco_transforms = get_data_transforms(dataset)
    if dataset == 'cifar10':
        if moco:
            test_dataset = CIFAR10_idxed(data_dir, train=False, transform=TwoCropTransform(moco_transforms))
        else:
            test_dataset = CIFAR10_idxed(data_dir, train=False, transform=test_transforms)
    elif dataset == 'cifar100':
        if moco:
            test_dataset = CIFAR100_idxed(data_dir, train=False, transform=TwoCropTransform(moco_transforms))
        else:
            test_dataset = CIFAR100_idxed(data_dir, train=False, transform=test_transforms)
    elif dataset == 'imagenet':
        test_dataset = IMAGENET_idxed(data_dir, split='val', transform=test_transforms)

    test_dataset.data = test_data
    test_dataset.targets = test_label

    if exclude_indices is None:
        train_indices = []
        for i in np.unique(test_label):
            index_i = np.where(test_label == i)[0]
            perm = np.random.permutation(index_i)
            len_train = int(len(index_i) * train_val_split)
            train_indices.extend(perm[:len_train])
    else:
        all_indices = np.arange(len(test_data))
        train_indices = list(np.setdiff1d(all_indices, exclude_indices))
    train_dataset = data.Subset(test_dataset, train_indices)

    train_loader = data.DataLoader(train_dataset, batch_size=n_bat, shuffle=shuffle,
                                   num_workers=workers, pin_memory=pin_memory, drop_last=drop_last)
    test_loader = data.DataLoader(test_dataset, batch_size=n_bat, shuffle=shuffle,
                                 num_workers=workers, pin_memory=pin_memory, drop_last=drop_last)
    return train_loader, test_loader


