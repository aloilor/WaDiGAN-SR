import argparse
import os
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from datasets_prep.dataset import create_dataset
from torch.multiprocessing import Process


if __name__ == '__main__':
    device = torch.device('cuda:{}'.format(gpu))
    batch_size = 32


    dataroot = "C:\\Users\\Lorenzo\\Desktop\\thesis\\datasets\\celebahq_16_128\\"
    save_dir = "C:\\Users\\Lorenzo\\Desktop\\thesis\\datasets\\train_set\\"
    print (save_dir)

    dataset = create_dataset(dataroot,
                            l_resolution=16,
                            r_resolution=128,
                            split="train",
                            data_len=-1,
                            need_LR=False
                            )

    train_size = int(0.95 * len(dataset))  # 95% for training
    test_size = len(dataset) - train_size  # 5% for testing
    
    if rank == 0:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    # Set a seed for reproducibility
    torch.manual_seed(42) 
    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_data_loader = torch.utils.data.DataLoader(train_set,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=3,
                                            pin_memory=True,
                                            drop_last=False)


    idx = 0
    for sample in enumerate (train_data_loader):

        hr = data_dict['HR'] # hr_img
        sample_index = data_dict['Index']

        hr = hr.to(device, non_blocking=True)

        for x in enumerate (hr):
            torchvision.utils.save_image(x, '{}{}.png'.format(save_dir, idx))
            idx += 1
        if idx % 100 == 0:
            print("Saved the first {} images".format(idx))
        