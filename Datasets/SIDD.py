import os
import random
import numpy as np
import scipy.io as sio
from PIL import Image
from glob import glob
from natsort import natsorted

import torch
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, DataLoader

from .utils import *

import ipdb

# This class is for SIDD_Medium train on sRGB patches
from natsort import natsorted
from glob import glob
class SIDD_sRGB_Train_DataLoader(Dataset):
    def __init__(self, path, length, patch_size, patched_input=False):
        super(SIDD_Medium_sRGB_Train_DataLoader, self).__init__()
        assert length % 160 == 0, 'You should specify a proper length.'

        self.len            = length
        self.patch_size     = patch_size
        self.patched_input  = patched_input
        self.noisy_imgs, self.clean_imgs = [], []
        self.transform      = torchvision.transforms.ToTensor()
        self.return_name    = False

        if not self.patched_input:
            # get file names from original dataset
            imgs = natsorted(glob(os.path.join(path, '*', '*.PNG')))
            for img in imgs:
                img_name = os.path.split(img)[-1]
                if 'GT' in img_name:
                    self.clean_imgs.append(img)
                if 'NOISY' in img_name:
                    self.noisy_imgs.append(img)
        else:
            # get file names from cropped dataset
            self.noisy_imgs = natsorted(glob(os.path.join(path, 'input', '*.png')))
            self.clean_imgs = natsorted(glob(os.path.join(path, 'target', '*.png')))

    def __len__(self):
        return self.len if self.len else len(self.noisy_imgs)

    def __getitem__(self, index):
        index = index % len(self.noisy_imgs) # in case out of range.
        aug   = random.randint(0, 7)

        # read image
        noisy_img   = np.array(Image.open(self.noisy_imgs[index]))
        clean_img   = np.array(Image.open(self.clean_imgs[index]))
        
        # random crop
        if self.patch_size:
            noisy_patch, clean_patch = crop_patches(noisy_img, clean_img,
                                                    self.patch_size)

        # random augmentation
        noisy_patch = Data_Augmentation(self.transform(noisy_patch), aug)
        clean_patch = Data_Augmentation(self.transform(clean_patch), aug)

        filename = os.path.splitext(os.path.split(self.clean_imgs[index])[-1])[0]
        if self.return_name:
            return noisy_patch, clean_patch, filename
        else:
            return noisy_patch, clean_patch

# this class is for SIDD val on sRGB patches.
# could download patched val data from https://drive.google.com/drive/folders/1S44fHXaVxAYW3KLNxK41NYCnyX9S79su
# If the full dataset is found, this class could be extended to support full graph/any patched size input.
class SIDD_sRGB_Val_DataLoader(Dataset):
    def __init__(self, path, patch_size=None):
        super(SIDD_sRGB_Val_DataLoader, self).__init__()
        self.noisy_imgs = natsorted(glob(os.path.join(path, 'input', '*.png')))
        self.clean_imgs = natsorted(glob(os.path.join(path, 'target', '*.png')))
        self.length     = len(self.noisy_imgs)
        self.patch_size = patch_size
        self.transform  = torchvision.transforms.ToTensor()
        self.return_name= False

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index %= self.length

        noisy_patch = np.array(Image.open(self.noisy_imgs[index]))
        clean_patch = np.array(Image.open(self.clean_imgs[index]))

        if self.patch_size:
            noisy_patch = F.center_crop(noisy_patch, (self.patch_size, self.patch_size))
            clean_patch = F.center_crop(clean_patch, (self.patch_size, self.patch_size))

        noisy_patch, clean_patch = self.transform(noisy_patch), \
                                   self.transform(clean_patch)
        filename = os.path.splitext(os.path.split(self.clean_imgs[index])[-1])[0]

        if self.return_name:
            return noisy_patch, clean_patch, filename
        else:
            return noisy_patch, clean_patch

# this class is for SIDD test on sRGB patches stored in .mat format
class SIDD_sRGB_mat_Test_DataLoader(Dataset):
    """
    Args:
        path - the dir to '*.mat' file
    """
    def __init__(self, path):
        super(SIDD_sRGB_mat_Test_DataLoader, self).__init__()
        self.len = 1280
        noisy_patch  = sio.loadmat(os.path.join(path, 'ValidationNoisyBlocksSrgb.mat'))
        target_patch = sio.loadmat(os.path.join(path, 'ValidationGtBlocksSrgb.mat'))
        noisy_array  = np.float32(np.array(noisy_patch['ValidationNoisyBlocksSrgb'])) 
        assert noisy_array.shape == (40, 32, 256, 256, 3)
        noisy_array  = np.resize(noisy_array / 255., (self.len, 256, 256, 3))
        target_array = np.float32(np.array(target_patch['ValidationGtBlocksSrgb']))
        target_array = np.resize(target_array / 255., (self.len, 256, 256, 3))

        self.transform    = torchvision.transforms.ToTensor()
        self.noise_patch  = noisy_array
        self.target_patch = target_array

    def __getitem__(self, index):
        input_patch = self.noise_patch[index]
        target_patch = self.target_patch[index]

        return self.transform(input_patch), self.transform(target_patch)

    def __len__(self):
        return self.len


