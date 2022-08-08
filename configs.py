import os
import sys

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import args
import vars as v
import models


def _get_cifar_10_dataloaders():
    _backup_print = sys.stdout
    sys.stdout = open(os.devnull, "w")
    norm_params = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*norm_params),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(*norm_params),
        ]
    )
    trainset = datasets.CIFAR10(
        root=args.data_path, train=True, download=True, transform=transform_train
    )
    trainloader = DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    testset = datasets.CIFAR10(
        root=args.data_path, train=False, download=True, transform=transform_test
    )
    testloader = DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    sys.stdout = _backup_print

    v.trainloader = trainloader
    v.testloader = testloader


def basic_resnet_18_cifar_10():
    v.model = models.ResNet_18()
    _get_cifar_10_dataloaders()
