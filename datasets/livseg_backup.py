import numpy as np
import random
from skimage.io import imread
from skimage import color, transform

import torch
import torch.utils.data
from torchvision import datasets, models, transforms



from torch.utils.data import random_split, DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.datasets import MNIST
from glob import glob
import os
import torch

class LivSegDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, img_paths, mask_paths):
        self.cfg = cfg
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    
    def __len__(self):
        return len(self.img_paths)
    
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        
        npimage = np.load(img_path)
        
        npmask = np.load(mask_path)
        
        npimage = npimage.transpose((2, 0, 1))

        liver_label = npmask.copy()
        liver_label[npmask == 2] = 1
        liver_label[npmask == 1] = 1

        tumor_label = npmask.copy()
        tumor_label[npmask == 1] = 0
        tumor_label[npmask == 2] = 1

        nplabel = np.empty((448,448,2))
        nplabel[:, :, 0] = liver_label
        nplabel[:, :, 1] = tumor_label
        nplabel = nplabel.transpose((2, 0, 1))

        nplabel = nplabel.astype("float32")
        npimage = npimage.astype("float32")
        #sample = {'image': npimage, 'mask': nplabel}

        if self.cfg.transformation:
            npimage = self.transform(npimage)
        

        return npimage, nplabel

def get_liverseg(cfg,dataset_name):
    
    if dataset_name=='ircad':
        img_paths = sorted(glob(os.path.join(cfg.ircad_data_path,"*",'img*')))
        mask_paths = sorted(glob(os.path.join(cfg.ircad_data_path,"*",'mask*')))
    elif dataset_name=='lits':
        img_paths = sorted(glob(os.path.join(cfg.lits_data_path,"*",'img*')))
        mask_paths = sorted(glob(os.path.join(cfg.lits_data_path,"*",'mask*')))
    elif dataset_name=='sliver':
        img_paths = sorted(glob(os.path.join(cfg.sliver_data_path,"*",'img*')))
        mask_paths = sorted(glob(os.path.join(cfg.sliver_data_path,"*",'mask*')))    
    
    dataset_samples = LivSegDataset(cfg, img_paths, mask_paths)
    
    num_total = len(dataset_samples)
    num_test = int(cfg.test_ratio*num_total)
    num_train = num_total - num_test
    training_set, test_set = random_split(dataset_samples, [num_train, num_test], torch.Generator().manual_seed(2023))
    return training_set, test_set

def prepare_dataset(cfg):
    training_set, test_set = get_liverseg(cfg)
    
    #split trainset into 'num_partitions' trainset
    num_image = len(training_set) // cfg.num_partitions
    
    partition_len = [num_image]*cfg.num_partitions
    
    for i in range(len(training_set)-sum(partition_len)):
        partition_len[i] += 1
    
    trainsets = random_split(training_set,partition_len, torch.Generator().manual_seed(2023))
    
    #create dataoaders with train+val support
    trainloaders= []
    valloaders = []
    for trainset_ in trainsets:
        num_total = len(trainset_)
        num_val = int(cfg.val_ratio*num_total)
        num_train = num_total - num_val

        for_train, for_val = random_split(trainset_, [num_train, num_val], torch.Generator().manual_seed(2023))
        
        trainloaders.append(DataLoader(for_train, batch_size=cfg.batch_size, shuffle=True))
        valloaders.append(DataLoader(for_val, batch_size=cfg.batch_size, shuffle=False))
    
    testloader = DataLoader(test_set, batch_size=cfg.batch_size)
    return trainloaders, valloaders, testloader


def prepare_multiple_datasets(cfg):
    
    ircad_training_set, ircad_test_set = get_liverseg(cfg, "ircad")
    lits_training_set, lits_test_set = get_liverseg(cfg, "lits")
    #sliver_training_set, sliver_test_set = get_liverseg(cfg, "sliver")

    #test_set = ircad_test_set + lits_test_set + sliver_test_set
    test_set = ircad_test_set + lits_test_set
    print("test set size")
    print(len(ircad_test_set))
    print(len(lits_test_set))
    #print(len(sliver_test_set))

    #print("training set size")
    #print(len(ircad_training_set))
    #print(len(lits_training_set))
    #print(len(sliver_training_set))
    
    #split trainset into 'num_partitions' trainset
    ircad_num_image = len(ircad_training_set) // cfg.num_partitions
    lits_num_image = len(lits_training_set) // cfg.num_partitions
    #sliver_num_image = len(sliver_training_set) // cfg.num_partitions
    
    ircad_partition_len = [ircad_num_image]*cfg.num_partitions
    lits_partition_len = [lits_num_image]*cfg.num_partitions
    #sliver_partition_len = [sliver_num_image]*cfg.num_partitions
    
    for i in range(len(ircad_training_set)-sum(ircad_partition_len)):
        ircad_partition_len[i] += 1
    
    for i in range(len(lits_training_set)-sum(lits_partition_len)):
        lits_partition_len[i] += 1
    
    #for i in range(len(sliver_training_set)-sum(sliver_partition_len)):
    #    sliver_partition_len[i] += 1
    
    ircad_trainsets = random_split(ircad_training_set,ircad_partition_len, torch.Generator().manual_seed(2023))
    lits_trainsets = random_split(lits_training_set,lits_partition_len, torch.Generator().manual_seed(2023))
    #sliver_trainsets = random_split(sliver_training_set,sliver_partition_len, torch.Generator().manual_seed(2023))
    
    #trainsets = [lits_trainsets,ircad_trainsets, sliver_trainsets]
    trainsets = [lits_trainsets,ircad_trainsets]
    print("trainset clients num: ", str(len(trainsets)))

    #create dataoaders with train+val support
    trainloaders= []
    valloaders = []
    for trainset in trainsets:
        for set_ in trainset:
            num_total = len(set_)
            num_val = int(cfg.val_ratio*num_total)
            num_train = num_total - num_val

            for_train, for_val = random_split(set_, [num_train, num_val], torch.Generator().manual_seed(2023))
        
        trainloaders.append(DataLoader(for_train, batch_size=cfg.batch_size, shuffle=True))
        valloaders.append(DataLoader(for_val, batch_size=cfg.batch_size, shuffle=False))
    
    
    testloader = DataLoader(test_set, batch_size=cfg.batch_size)
    return trainloaders, valloaders, testloader

