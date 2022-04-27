# -*- coding: utf-8 -*-

import sys

from torchvision import transforms


def get_data_transform(data: str):
    if data == 'cifar':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),   # input arguments: length&width of a figure
            #transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # convert PIL image or numpy.ndarray to tensor
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        test_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    else:
        print('Data must be {} or {} !'.format('mnist', 'cifar'))
        sys.exit(-1)

    return train_transform, test_transform
