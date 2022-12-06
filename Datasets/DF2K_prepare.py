import os
import shutil
from glob import glob
from natsort import natsorted

DIV2K_path = './DIV2K'
DIV2K_HR_path = os.path.join(DIV2K_path, 'DIV2K_train_HR')
DIV2K_LR_path = os.path.join(DIV2K_path, 'DIV2K_train_LR_bicubic')
Flickr_path = './Flickr2K'
Flickr_HR_path = os.path.join(Flickr_path, 'Flickr2K_HR')
Flickr_LR_path = os.path.join(Flickr_path, 'Flickr2K_LR_bicubic')
DF2K_path = './DF2K'
DF2K_HR_path = os.path.join(DF2K_path, 'DF2K_train_HR')
DF2K_LR_path = os.path.join(DF2K_path, 'DF2K_train_LR_bicubic')

div2k_hr_imgs = natsorted(glob(os.path.join(DIV2K_HR_path, '*.png')))
flickr_hr_imgs = natsorted(glob(os.path.join(Flickr_HR_path, '*.png')))

if not os.path.exists(DF2K_path):
    os.mkdir(DF2K_path)
    if not os.path.exists(DF2K_HR_path):
        os.mkdir(DF2K_HR_path)
    if not os.path.exists(DF2K_LR_path):
        os.mkdir(DF2K_LR_path)

# copy HR imgs
if len(natsorted(glob(os.path.join(DF2K_HR_path, '*.png')))) < 900:
    for img in div2k_hr_imgs:
        img_name = img.split('/')[-1]
        shutil.copyfile(img, os.path.join(DF2K_HR_path, img_name))
if len(natsorted(glob(os.path.join(DF2K_HR_path, '*.png')))) < 3550:
    for img in flickr_hr_imgs:
        origin_name = img.split('/')[-1]
        split_name = origin_name[-8:-4]
        img_idx = int(split_name) + 900
        save_name = str(img_idx)
        if len(save_name) == 3:
            save_name = '0' + save_name
        save_name += '.png'
        shutil.copyfile(img, os.path.join(DF2K_HR_path, save_name))

# copy LR imgs
for scale in ['X2', 'X3', 'X4']:
    LR_dir = os.path.join(DF2K_LR_path, scale)
    if not os.path.exists(LR_dir):
        os.mkdir(LR_dir)
    div2k_lr_imgs = natsorted(glob(os.path.join(DIV2K_LR_path, scale, '*.png')))
    if len(natsorted(glob(os.path.join(LR_dir, '*.png')))) < 900:
        for img in div2k_lr_imgs:
            img_name = img.split('/')[-1]
            shutil.copyfile(img, os.path.join(LR_dir, img_name))
    flickr_lr_imgs = natsorted(glob(os.path.join(Flickr_LR_path, scale, '*.png')))
    if len(natsorted(glob(os.path.join(LR_dir, '*.png')))) < 3550:
        for img in flickr_lr_imgs:
            img_name = img.split('/')[-1]
            save_idx = str(int(img_name[2:6]) + 900)
            save_name = save_idx + 'x' + scale[-1] + '.png'
            if len(save_idx) == 3:
                save_name = '0' + save_name
            shutil.copyfile(img, os.path.join(LR_dir, save_name))


