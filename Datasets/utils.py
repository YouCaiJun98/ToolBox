import random
import numpy as np

# randomly crop a patch from image
def crop_patch(im, pch_size):
    H = im.shape[0]
    W = im.shape[1]
    rnd_H = random.randint(0, H-pch_size)
    rnd_W = random.randint(0, W-pch_size)
    pch = im[rnd_H:rnd_H + pch_size, rnd_W:rnd_W + pch_size]
    return pch

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