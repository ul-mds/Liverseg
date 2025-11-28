import numpy as np
import random
from skimage.io import imread
from skimage import color, transform

import torch
import torch.utils.data
from torch.utils.data import random_split, DataLoader

from torchvision import datasets, models, transforms
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.datasets import MNIST
from glob import glob
import os
import pickle

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
    
    #if dataset_name=='ircad':
    #    print(cfg.ircad_data_path)
    #    img_paths = sorted(glob(os.path.join(cfg.ircad_data_path,"*",'img*')))
    #    mask_paths = sorted(glob(os.path.join(cfg.ircad_data_path,"*",'mask*')))
    #elif dataset_name=='lits':
    #    img_paths = sorted(glob(os.path.join(cfg.lits_data_path,"*",'img*')))
    #    mask_paths = sorted(glob(os.path.join(cfg.lits_data_path,"*",'mask*')))
    #elif dataset_name=='sliver':
    #    img_paths = sorted(glob(os.path.join(cfg.sliver_data_path,"*",'img*')))
    #    mask_paths = sorted(glob(os.path.join(cfg.sliver_data_path,"*",'mask*')))    
    
    #img_paths = img_paths[:120]
    #mask_paths = mask_paths[:120]
    #dataset_samples = LivSegDataset(cfg, img_paths, mask_paths)
    
    #num_total = len(dataset_samples)
    #num_test = int(cfg.test_ratio*num_total)
    #num_train = num_total - num_test
    #training_set, test_set = random_split(dataset_samples, [num_train, num_test], torch.Generator().manual_seed(2023))
    if dataset_name=='ircad':
        print(cfg.ircad_data_path)
        training_img_paths = sorted(glob(os.path.join(cfg.ircad_data_path,"training","*",'img*')))
        training_mask_paths = sorted(glob(os.path.join(cfg.ircad_data_path,"training","*",'mask*')))

        #val_img_paths = sorted(glob(os.path.join(cfg.ircad_data_path,"val","*",'img*')))
        #val_mask_paths = sorted(glob(os.path.join(cfg.ircad_data_path,"val","*",'mask*')))

        test_img_paths = sorted(glob(os.path.join(cfg.ircad_data_path,"test","*",'img*')))
        test_mask_paths = sorted(glob(os.path.join(cfg.ircad_data_path,"test","*",'mask*')))
	
        image_set = training_img_paths + test_img_paths
        mask_set = training_mask_paths + test_mask_paths

    elif dataset_name=='lits':
        img_paths = sorted(glob(os.path.join(cfg.lits_data_path,"*",'img*')))
        mask_paths = sorted(glob(os.path.join(cfg.lits_data_path,"*",'mask*')))


        training_img_paths = sorted(glob(os.path.join(cfg.lits_data_path,"training","*",'img*')))
        training_mask_paths = sorted(glob(os.path.join(cfg.lits_data_path,"training","*",'mask*')))

        #val_img_paths = sorted(glob(os.path.join(cfg.lits_data_path,"val","*",'img*')))
        #val_mask_paths = sorted(glob(os.path.join(cfg.lits_data_path,"val","*",'mask*')))

        test_img_paths = sorted(glob(os.path.join(cfg.lits_data_path,"test","*",'img*')))
        test_mask_paths = sorted(glob(os.path.join(cfg.lits_data_path,"test","*",'mask*')))
    	
        image_set = training_img_paths + test_img_paths
        mask_set = training_mask_paths + test_mask_paths
    #training_set = LivSegDataset(cfg, training_img_paths, training_mask_paths)
    #val_set = LivSegDataset(cfg, val_img_paths, val_mask_paths)
    #test_set = LivSegDataset(cfg, test_img_paths, test_mask_paths)
    sets = LivSegDataset(cfg, image_set, mask_set)
    #return training_set, test_set
    return sets
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
    
    #ircad_training_set, ircad_test_set = get_liverseg(cfg, "ircad")
    #lits_training_set, lits_test_set = get_liverseg(cfg, "lits")
    #sliver_training_set, sliver_test_set = get_liverseg(cfg, "sliver")
    
    #ircad_image_sets = get_liverseg(cfg, "ircad")
    #lits_image_sets = get_liverseg(cfg, "lits")
    
    #lits_num_total = len(lits_image_sets)
    #lits_num_test = int(0.2*lits_num_total)
    #lits_num_train = lits_num_total - lits_num_test
    
    #ircad_num_total = len(ircad_image_sets)
    #ircad_num_test = int(0.2*ircad_num_total)
    #ircad_num_train = ircad_num_total - ircad_num_test
    
    #lits_training_set, lits_test_set = random_split(lits_image_sets, [lits_num_train, lits_num_test], torch.Generator().manual_seed(2023))
    #ircad_training_set, ircad_test_set = random_split(ircad_image_sets, [ircad_num_train, ircad_num_test], torch.Generator().manual_seed(2023))
    
    #with open('lits_training.pkl','wb') as f: pickle.dump(lits_training_set, f)
    #with open('lits_test.pkl','wb') as f: pickle.dump(lits_test_set, f)
    
    #with open('ircad_training.pkl','wb') as f: pickle.dump(ircad_training_set, f)
    #with open('ircad_test.pkl','wb') as f: pickle.dump(ircad_test_set, f)
    
    lits_training_set, lits_test_set = pickle.load(open('lits_training.pkl', 'rb')), pickle.load(open('lits_test.pkl', 'rb'))
    ircad_training_set, ircad_test_set = pickle.load(open('ircad_training.pkl', 'rb')), pickle.load(open('ircad_test.pkl', 'rb'))
    
    #all_test_set = ircad_test_set + lits_test_set + sliver_test_set
    #all_test_set = ircad_test_set + lits_test_set
    all_test_set = lits_test_set + ircad_test_set
    print("test set size")
    print(len(ircad_test_set))
    print(len(lits_test_set))
    #print(len(sliver_test_set))
    print(len(all_test_set))

    print("training set size")
    print(len(ircad_training_set))
    print(len(lits_training_set))
    #print(len(sliver_training_set))
    
    #trainsets = [lits_training_set,ircad_training_set, sliver_training_set]
    #testsets = [lits_test_set, ircad_test_set, sliver_test_set, all_test_set]
    #trainsets = [ircad_training_set, lits_training_set]
    testsets = [lits_test_set, ircad_test_set, all_test_set]
    
    #valsets = [lits_val_set, ircad_val_set]
    #trainsets = [lits_training_set, ircad_training_set]
    trainsets = [lits_training_set, ircad_training_set]
    print("trainset clients num: ", str(len(trainsets)))

    #create dataoaders with train+val support
    trainloaders= []
    #valloaders = []

    testloaders= []
    
    for i,_ in enumerate(trainsets):
        #num_total = len(trainsets[i])
        #num_val = int(cfg.val_ratio*num_total)
        #num_train = num_total - num_val

        #for_train, for_val = random_split(trainsets[i], [num_train, num_val], torch.Generator().manual_seed(2023))
        
        trainloaders.append(DataLoader(trainsets[i], batch_size=cfg.batch_size, shuffle=True))
        #valloaders.append(DataLoader(for_val, batch_size=cfg.batch_size, shuffle=True))
    
    for testset in testsets:
        testloaders.append(DataLoader(testset, batch_size=cfg.batch_size, shuffle=False))
        
    return trainloaders, testloaders
