import os

import rawpy
import numpy as np
import PIL.Image as Image

import torch
from torch.utils import data

__all__ = ['SID_Sony', 'SID_Fuji']


# Read meta information of the raw images from their filenames.
def metainfo(image_list_file):
    dict_list = []
    with open(image_list_file, 'r') as f:
        for i, img_pair in enumerate(f):
            img_pair = img_pair.strip()
            input_file, target_file, iso, focus = img_pair.split(' ')
            # get exposure time
            read_expo = lambda x: float(os.path.split(x)[-1][9:-5])
            input_expo  = read_expo(input_file)
            target_expo = read_expo(target_file)
            ratio = min(target_expo/input_expo, 300)
            dict_list.append({
                'input':       input_file,
                'target':      target_file,
                'input_expo':  input_expo,
                'target_expo': target_expo,
                'ratio':       ratio,
                'iso':         iso,
                'focus':       focus,
            })
    return dict_list

class Raw_Base(data.Dataset):
    def __init__(self, root, image_list_file, patch_size=None, stage_in='raw', stage_out='raw',
                 data_aug=True, gt_png=False):
        """
        :param root: dataset directory
        :param image_list_file: contains image file names under root
        :param patch_size: if None, full images are returned, otherwise patches are returned.
                           should note that the target patch is cropped AFTER packing.
        :param stage_in:  type of input images for the model. should be raw or sRGB.
        :param stage_out: type of target images for the model. should be raw or sRGB.
        :param gt_png: whether use the preprocessed png images.
        """
        assert os.path.exists(root), "Root: {} not found.".format(root)
        assert os.path.exists(image_list_file), \
            "image_list_file: {} not found.".format(image_list_file)
        assert stage_in  in ['raw', 'sRGB'], \
            "Input image type {} should be raw or sRGB.".format(stage_in)
        assert stage_out in ['raw', 'sRGB'], \
            "Target image type {} sholud be raw or sRGB".format(stage_out)

        self.root = root
        self.image_list_file = image_list_file
        self.stage_in  = stage_in
        self.stage_out = stage_out
        self.gt_png = gt_png
        self.patch_size = patch_size
        self.data_aug = data_aug

        self.img_info = metainfo(self.image_list_file)

    def __getitem__(self, index):
        """
        When sRGB images are needed, we just adopt the process procedure of Rawpy,
        There are more complicated ISP - https://github.com/Vandermode/ELD/blob/master/util/process.py
        """
        info = self.img_info[index]
        input_file, target_file = info['input'], info['target']
        # input image
        input_raw = rawpy.imread(os.path.join(self.root, input_file))
        # raw input
        if self.stage_in == 'raw':
            # Please note that we PACK the input raw images by convention. 
            input_img = self.pack_raw(input_raw) * info['ratio']
        # sRGB input
        elif self.stage_in == 'sRGB':
            im = input_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            input_img = np.transpose(np.float32(im / 65535.), (2, 0, 1))

        # target image
        if not self.gt_png:
            target_raw = rawpy.imread(os.path.join(self.root, target_file))
        # raw target
        if self.stage_out == 'raw':
            # Note that we pack the output raw images for simple calculation.
            target_img = self.pack_raw(target_raw)
        # sRGB target
        elif self.stage_out == 'sRGB':
            if self.gt_png:
                target_img = np.array(Image.open(os.path.join(self.root, target_file)), dtype=np.float32)
                target_img = np.transpose((target_img / 255.), (2, 0, 1))
            else:
                im = target_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
                target_img = np.transpose(np.float32(im / 65535.), (2, 0, 1))

        if self.patch_size:
            H, W = input_img.shape[1:3]
            ps = self.patch_size
            assert W-ps>0 and H-ps>0, "The patch size is larger than the current img."
            xx = np.random.randint(0, W-ps)
            yy = np.random.randint(0, H-ps)
            if self.stage_in == self.stage_out: # raw2raw or rgb2rgb
                input_img  = input_img[:, yy:yy + ps, xx:xx + ps]
                target_img = target_img[:,yy:yy + ps, xx:xx + ps]
            elif self.stage_in == 'raw' and self.stage_out == 'sRGB': # raw2rgb
                input_img  = input_img[:, yy:yy + ps, xx:xx + ps]
                target_img = target_img[:, yy*2:(yy+ps)*2, xx*2:(xx+ps)*2]

        if self.data_aug:
            if np.random.randint(2, size=1)[0] == 1:  # random flip
                input_img  = np.flip(input_img, axis=1) # H
                target_img = np.flip(target_img, axis=1)
            if np.random.randint(2, size=1)[0] == 1:
                input_img  = np.flip(input_img, axis=2) # W
                target_img = np.flip(target_img, axis=2)
            if np.random.randint(2, size=1)[0] == 1:  # random transpose
                input_img  = np.transpose(input_img, (0, 2, 1))
                target_img = np.transpose(target_img, (0, 2, 1))

        input_img  = np.maximum(np.minimum(input_img, 1.0), 0)
        input_img  = torch.from_numpy(np.ascontiguousarray(input_img)).float()
        target_img = torch.from_numpy(np.ascontiguousarray(target_img)).float()

        return input_img, target_img

    def __len__(self):
        return len(self.img_info)

class SID_Sony(Raw_Base):
    def __init__(self, root, image_list_file, patch_size=None, stage_in='raw',
                 stage_out='raw', data_aug=True, gt_png=False):
        super(SID_Sony, self).__init__(root, image_list_file, patch_size=patch_size, stage_in=stage_in,
                                       stage_out=stage_out, data_aug=data_aug, gt_png=gt_png)

    def pack_raw(self, raw):
        # pack Bayer image to 4 channels
        im = raw.raw_image_visible.astype(np.float32)
        raw_pattern = raw.raw_pattern
        white_level = raw.white_level
        black_level = np.array(raw.black_level_per_channel)[:,None,None].astype(np.float32)
        img_shape = im.shape
        H = img_shape[0]
        W = img_shape[1]
        ch_idx = lambda x: np.where(raw_pattern == x)
        R, G1, B, G2 = ch_idx(0), ch_idx(1), ch_idx(2), ch_idx(3)

        # Through this method, different Bayer Patterns could be packed as the same
        # order.
        packed_raw = np.stack((im[R [0][0]:H:2,  R[1][0]:W:2],
                               im[G1[0][0]:H:2, G1[1][0]:W:2],
                               im[B [0][0]:H:2,  B[1][0]:W:2],
                               im[G2[0][0]:H:2, G2[1][0]:W:2]), axis=0)
        packed_raw = (packed_raw - black_level) / (white_level - black_level)
        return np.clip(packed_raw, 0, 1)

class SID_Sony_patched(SID_Sony):
    def __init__(self, root, image_list_file, patch_size=None, stage_in='raw',
                 stage_out='raw', data_aug=True, gt_png=False):
        super(SID_Sony, self).__init__(root, image_list_file, patch_size=patch_size, stage_in=stage_in,
                                       stage_out=stage_out, data_aug=data_aug, gt_png=gt_png)




class SID_Fuji(Raw_Base):
    def __init__(self, root, image_list_file, patch_size=None, stage_in='raw',
                 stage_out='raw', data_aug=True, gt_png=False):
        super(SID_Fuji, self).__init__(root, image_list_file, patch_size=patch_size, stage_in=stage_in, 
                                       stage_out=stage_out, data_aug=data_aug, gt_png=gt_png)

    def pack_raw(self, raw):
        # pack X-Trans image to 9 channels
        # the process below comes from https://github.com/cydonia999/Learning_to_See_in_the_Dark_PyTorch/blob/master/datasets/raw_images.py
        im = raw.raw_image_visible.astype(np.float32)
        raw_pattern = raw.raw_pattern
        white_level = raw.white_level
        img_shape = im.shape

        im = np.maximum(im - 1024, 0) / (white_level - 1024)  # subtract the black level
        H = (img_shape[0] // 6) * 6
        W = (img_shape[1] // 6) * 6

        packed_raw = np.zeros((H // 3, W // 3, 9))

        # 0 R
        packed_raw[0::2, 0::2, 0] = im[0:H:6, 0:W:6]
        packed_raw[0::2, 1::2, 0] = im[0:H:6, 4:W:6]
        packed_raw[1::2, 0::2, 0] = im[3:H:6, 1:W:6]
        packed_raw[1::2, 1::2, 0] = im[3:H:6, 3:W:6]

        # 1 G
        packed_raw[0::2, 0::2, 1] = im[0:H:6, 2:W:6]
        packed_raw[0::2, 1::2, 1] = im[0:H:6, 5:W:6]
        packed_raw[1::2, 0::2, 1] = im[3:H:6, 2:W:6]
        packed_raw[1::2, 1::2, 1] = im[3:H:6, 5:W:6]

        # 1 B
        packed_raw[0::2, 0::2, 2] = im[0:H:6, 1:W:6]
        packed_raw[0::2, 1::2, 2] = im[0:H:6, 3:W:6]
        packed_raw[1::2, 0::2, 2] = im[3:H:6, 0:W:6]
        packed_raw[1::2, 1::2, 2] = im[3:H:6, 4:W:6]

        # 4 R
        packed_raw[0::2, 0::2, 3] = im[1:H:6, 2:W:6]
        packed_raw[0::2, 1::2, 3] = im[2:H:6, 5:W:6]
        packed_raw[1::2, 0::2, 3] = im[5:H:6, 2:W:6]
        packed_raw[1::2, 1::2, 3] = im[4:H:6, 5:W:6]

        # 5 B
        packed_raw[0::2, 0::2, 4] = im[2:H:6, 2:W:6]
        packed_raw[0::2, 1::2, 4] = im[1:H:6, 5:W:6]
        packed_raw[1::2, 0::2, 4] = im[4:H:6, 2:W:6]
        packed_raw[1::2, 1::2, 4] = im[5:H:6, 5:W:6]

        packed_raw[:, :, 5] = im[1:H:3, 0:W:3]
        packed_raw[:, :, 6] = im[1:H:3, 1:W:3]
        packed_raw[:, :, 7] = im[2:H:3, 0:W:3]
        packed_raw[:, :, 8] = im[2:H:3, 1:W:3]

        return packed_raw
