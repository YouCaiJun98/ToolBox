import glob
from PIL import Image

import torchvision
from torch.utils.data import Dataset

from .utils import *

__all__ = ['bsd68', 'bsd100', 'bsd200']

class BSD(Dataset):
    def __init__(self, gray:bool, pth, length, patch_size, sigma, rnd_aug):
        super(BSD, self).__init__()
        self.len = length
        self.patch_size = patch_size
        self.rnd_aug = rnd_aug
        self.sigma = sigma
        self.gray = gray

        rgb2gray = torchvision.transforms.Grayscale(1)
        self.images = []
        self.num_images = 0
        for p in glob.glob(pth + '*[jpg, png]'):
            img = Image.open(p)
            if self.gray:
                self.images.append(np.array(rgb2gray(img))[..., np.newaxis])
            else:
                self.images.append(np.array(img))

            self.num_images += 1

        self.transform = torchvision.transforms.ToTensor()

    def __getitem__(self, index):
        if self.len == len(self.images):
            # test setting
            idx = index
        else:
            # train setting
            idx = random.randint(0, self.num_images - 1)
        if self.patch_size:
            img =crop_patch(self.images[idx], self.patch_size)
        else:
            img = self.images[idx]

        # generate gaussian noise N(0, sigma^2)
        noise = np.random.randn(*(img.shape))
        noise_img = np.clip(img + noise * self.sigma, 0, 255).astype(np.uint8)

        aug_list = random_augmentation(img, noise_img) if self.rnd_aug else [img, noise_img]


        return self.transform(aug_list[1]), self.transform(aug_list[0])

    def __len__(self):
        return self.len
    
def bsd68(gray:bool, pth:str, length:int, patch_size:int, sigma:int, rnd_aug:bool):
    assert len(glob.glob(pth + '*[jpg, png]')) == 68, \
    'The target folder should contain 68 images for test.'
    dataset = BSD(gray, pth, length, patch_size, sigma, rnd_aug)
    return dataset

def bsd100(gray:bool, pth:str, length:int, patch_size:int, sigma:int, rnd_aug:bool):
    assert len(glob.glob(pth + '*[jpg, png]')) == 100, \
    'The target folder should contain 100 images for test.'
    dataset = BSD(gray, pth, length, patch_size, sigma, rnd_aug)
    return dataset

def bsd200(pth:str, gray:bool, length:int, sigma:int, rnd_aug:bool, patch_size:int=0):
    assert len(glob.glob(pth + '*[jpg, png]')) == 200, \
    'The target folder should contain 200 images for test or train.'
    dataset =  BSD(gray, pth, length, patch_size, sigma, rnd_aug)
    return dataset

