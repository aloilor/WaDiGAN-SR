import os

import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.datasets import CIFAR10, STL10

from .LRHR_dataset import LRHRDataset

def num_samples(dataset, train):
    if dataset == 'celeba':
        return 27000 if train else 3000
    elif dataset == 'ffhq':
        return 63000 if train else 7000
    else:
        raise NotImplementedError('dataset %s is unknown' % dataset)


def create_dataset(args):
    if args.dataset == 'celebahq_16_64':
        dataset = LRHRDataset(
                dataroot=args.datadir,
                datatype='lmdb',
                l_resolution=args.l_resolution,
                r_resolution=args.h_resolution,
                split="train",
                data_len=-1,
                need_LR=False
                )
        
    elif args.dataset == 'celebahq_16_128':
        dataset = LRHRDataset(
            dataroot=args.datadir,
            datatype='lmdb',
            l_resolution=args.l_resolution,
            r_resolution=args.h_resolution,
            split="train",
            data_len=-1,
            need_LR=False
            )
    elif args.dataset == 'div2k_128_512':
        dataset = LRHRDataset(
            dataroot=args.datadir,
            datatype='lmdb',
            l_resolution=args.l_resolution,
            r_resolution=args.h_resolution,
            split="train",
            data_len=-1,
            need_LR=False
            )
    elif args.dataset == 'df2k_128_512':
        dataset = LRHRDataset(
            dataroot=args.datadir,
            datatype='lmdb',
            l_resolution=args.l_resolution,
            r_resolution=args.h_resolution,
            split="train",
            data_len=-1,
            need_LR=False
            )


    return dataset
