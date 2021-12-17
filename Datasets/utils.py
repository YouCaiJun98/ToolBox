import random
import numpy as np

# randomly crop a patch from a gray image.
def crop_patch(im, patch_size, gray=False, rnd=None):
    H = im.shape[0]
    W = im.shape[1]

    H_pad = patch_size - H if H < patch_size else 0
    W_pad = patch_size - W if W < patch_size else 0
    if H_pad != 0 or W_pad != 0:
        im = np.pad(im, (0, 0, W_pad, H_pad), 'reflect')
        H = im.shape[0]
        W = im.shape[1]
    
    if rnd:
        (rnd_H, rnd_W) = rnd
    else:
        rnd_H = random.randint(0, H-patch_size)
        rnd_W = random.randint(0, W-patch_size)
    pch = im[rnd_H:rnd_H + patch_size, rnd_W:rnd_W + patch_size] if gray \
        else im[rnd_H:rnd_H + patch_size, rnd_W:rnd_W + patch_size, :]
    return pch

# randomly crop a pair of image patches in np format.
def crop_patches(im1, im2, patch_size, gray=False):
    assert im1.shape == im2.shape, 'input images should be of the same size.'
    H, W = im1.shape[0], im1.shape[1]

    H_pad = patch_size - H if H < patch_size else 0
    W_pad = patch_size - W if W < patch_size else 0
    if H_pad != 0 or W_pad != 0:
        im1 = np.pad(im1, (0, 0, W_pad, H_pad), 'reflect')
        im2 = np.pad(im2, (0, 0, W_pad, H_pad), 'reflect')
    rand = (random.randint(0, H-patch_size),
            random.randint(0, W-patch_size))

    patch1 = crop_patch(im1, patch_size, gray, rand)
    patch2 = crop_patch(im2, patch_size, gray, rand)
    
    return patch1, patch2

# Needs TorchVision Implementation
def data_augmentation(image, mode):
    if mode == 0:
        # original
        out = image
    elif mode == 1:
        # flip up and down
        out = np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(image)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(image, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(image, k=3)
        out = np.flipud(out)
    else:
        raise Exception('Invalid choice of image transformation')
    return out

# pytorch implemtation
import torch
def Data_Augmentation(img, mode):
    if   mode == 0:
        out = img
    elif mode == 1:
        out = img.flip(1)
    elif mode == 2:
        out = img.flip(2)
    elif mode == 3:
        out = torch.rot90(img, dims=(1,2))
    elif mode == 4:
        out = torch.rot90(img,dims=(1,2), k=2)
    elif mode == 5:
        out = torch.rot90(img,dims=(1,2), k=3)
    elif mode == 6:
        out = torch.rot90(img.flip(1),dims=(1,2))
    elif mode == 7:
        out = torch.rot90(img.flip(2),dims=(1,2))
    else:
        raise Exception('Invalid choice of image transformation')
    return out

def random_augmentation(*args):
    out = []
    flag_aug = random.randint(1,7)
    for data in args:
        out.append(data_augmentation(data, flag_aug).copy())
    return out

def inverse_augmentation(image, mode):
    if mode == 0:
        out = image
    elif mode == 1:
        out = np.flipud(image)
    elif mode == 2:
        out = np.rot90(image,k=-1)
    elif mode == 3:
        out = np.flipud(image)
        out = np.rot90(out, k=-1)
    elif mode == 4:
        out = np.rot90(image, k=-2)
    elif mode == 5:
        out = np.flipud(image)
        out = np.rot90(out, k=-2)
    elif mode == 6:
        out = np.rot90(image, k=-3)
    elif mode == 7:
        out = np.flipud(image)
        out = np.rot90(out, k=-3)
    else:
        raise Exception('Invalid choice of image transformation')
    return out

def image_show(x, title=None, cbar=False, figsize=None,
               converted_from_tensor=True, cmap=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    if converted_from_tensor:
        # in this case, x should be a np.ndarray
        x = x.transpose((2, 1, 0))
    plt.imshow(x, interpolation='nearest', cmap=cmap)
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()
