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

class GoPro_sRGB_Train_DataSet(Dataset):
    def __init__(self, path, patch_size, length=None):
        super(GoPro_sRGB_Train_DataSet, self).__init__()
        self.len           = length
        self.patch_size    = patch_size
        self.transform     = torchvision.transforms.ToTensor()
        self.return_name   = False

        self.noisy_imgs = natsorted(glob(os.path.join(path, 'input', '*.png')))
        self.clean_imgs = natsorted(glob(os.path.join(path, 'target', '*.png')))

    def __len__(self):
        return self.len if self.len else len(self.noisy_imgs)

    def __getitem__(self, index):
        index = index % self.__len__()
        rng   = random.randint(0, 7)

        # read image
        noisy_img = np.array(Image.open(self.noisy_imgs[index]))
        clean_img = np.array(Image.open(self.clean_imgs[index]))
        filename  = self.noisy_imgs[index].split('/')[-1][:-4]

        # random crop
        if self.patch_size:
            noisy_patch, clean_patch = crop_patches(noisy_img, clean_img, self.patch_size)

        # adjust gamma and saturation
        # Keep the same ratio as that of MPRNet
        if not random.randint(0, 2):
            noisy_patch = F.adjust_gamma(noisy_patch, 1)
            clean_patch = F.adjust_gamma(clean_patch, 1)

        if not random.randint(0, 2):
            sat_factor = 1 + (0.2 - 0.4*np.random.rand())
            noisy_patch = F.adjust_saturation(noisy_patch, sat_factor)
            clean_patch = F.adjust_saturation(clean_patch, sat_factor)

        # random augmentation
        noisy_patch = Data_Augmentation(self.transform(noisy_patch), aug)
        clean_patch = Data_Augmentation(self.transform(clean_patch), aug)

        if self.return_name:
            return noisy_patch, clean_patch, filename
        else:
            return noisy_patch, clean_patch



class GoPro_sRGB_Test_DataSet(Dataset):
    def __init__(self, path, patch_size=None):
        super(GoPro_sRGB_Test_DataSet, self).__init__()
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

        noisy_patch, clean_patch = self.transform(noisy_patch), \
                                   self.transform(clean_patch)

        if self.patch_size:
            noisy_patch = F.center_crop(noisy_patch, (self.patch_size, self.patch_size))
            clean_patch = F.center_crop(clean_patch, (self.patch_size, self.patch_size))

        filename = self.noisy_imgs[index].split('/')[-1][:-4]

        if self.return_name:
            return noisy_patch, clean_patch, filename
        else:
            return noisy_patch, clean_patch
