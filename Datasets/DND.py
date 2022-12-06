'''
Evaluation Script for DND dataset.
Should be conducted under the root path.
'''
import os
import cv2
import sys
import math
import time
import yaml
import argparse
import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm
import scipy.io as sio
from natsort import natsorted
from skimage import img_as_ubyte, img_as_float32

import torch
import torchvision
import torch.nn as nn
from   torchvision import transforms
import torchvision.transforms.functional as F
from   torch.utils.data import Dataset, DataLoader

import utils
import models

class DND_sRGB_Dataset(Dataset):
    def __init__(self, path):
        super(DND_sRGB_Dataset, self).__init__()
        self.imgs = natsorted(glob(os.path.join(path, 'input', '*.png')))
        self.len  = len(self.imgs)
        self.transform = transforms.ToTensor()

        for img in self.imgs:
            img_name = os.path.split(img)[-1]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        index %= self.len

        img = np.array(Image.open(self.imgs[index]))
        img = self.transform(img)
        imgname = os.path.splitext(os.path.split(self.imgs[index])[-1])[0]
        return img, imgname

def bundle_submission_srgb(folder, session):
    '''
    Bundles submission data for sRGB denoising

    submission_folder Folder where denoised images reside
    Output is written to <submission_folder>/bundled/. Please submit
    the content of this folder.
    '''
    out_folder = os.path.join(folder, session)
    try:
        os.mkdir(out_folder)
    except:
        pass
    israw = False
    eval_version = '1.0'

    for i in range(50):
        Idenoised = np.zeros((20,), dtype=np.object)
        for block in range(20):
            filename = '%04d_%d.mat'%(i+1, block+1)
            s = sio.loadmat(os.path.join(folder, filename))
            Idenoised[block] = s['Idenoised_crop']
        filename = '%04d.mat'%(i+1)
        sio.savemat(os.path.join(out_folder, filename),
                    {'Idenoised': Idenoised,
                     'israw': israw,
                     'eval_version': eval_version},
                    )

parser = argparse.ArgumentParser("☆ Denoising Evaluation on DND ☆")
parser.add_argument('-a', '--arch', metavar='ARCH', default='lsid')
parser.add_argument('--root', type=str, default='./datasets/DND/', help='root location of the dataset.')
parser.add_argument('--save_path', type=str, default='./dnd_results', help='parent path for saved dnd results.')
parser.add_argument('--gpu', type=str, help='gpu device ids.')
parser.add_argument('--resume', type=str, required=True,
                    help='checkpoint path of previous model.')
parser.add_argument('-c', '--configuration', required=True, help='model & validation settings.')
parser.add_argument('-d', '--debug', dest='save_flag', action='store_false',
                    help='if specified, results won\'t be saved.')
parser.add_argument('--save_img', action='store_true',
                    help='if specified, reconstructed png imgs will be saved.')



def main():
    args = parser.parse_args()

    # create save path
    if args.save_flag:
        # create dir for saving results
        save_name = '{}-DNDtest-{}'.format(args.arch, time.strftime("%Y%m%d-%H%M%S"))
        save_path = os.path.join(args.save_path, save_name)
        utils.create_exp_dir(save_path, scripts_to_save=[])
        os.mkdir(os.path.join(save_path, 'matfile'))
        os.mkdir(os.path.join(save_path, 'png'))

    # parse configurations
    with open(args.configuration, 'r') as rf:
        cfg = yaml.load(rf, Loader=yaml.FullLoader)
        model_cfg = cfg['model_settings']

    # get info logger
    args.evaluate = True
    logging = utils.get_logger(args)

    # set up device.
    if args.gpu and torch.cuda.is_available():
        args.gpu_flag = True
        device = torch.device('cuda')
        gpus = [int(d) for d in args.gpu.split(',')]
        torch.cuda.set_device(gpus[0]) # currently only single card is supported
        logging.info("Using GPU {}. Available gpu count: {}".format(gpus[0], torch.cuda.device_count()))
    else:
        args.gpu_flag = False
        device = torch.device('cpu')
        logging.info("\033[1;3mWARNING: Using CPU!\033[0m")

    # set up model.
    model = models.__dict__[args.arch](model_cfg)
    logging.info(model)
    if args.gpu_flag:
        model.cuda()
    else:
        logging.info("Using CPU. This will be slow.")

    # Resume a pretrained model.
    if args.resume:
        if os.path.isfile(args.resume):
            logging.info("Loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=device)

            if 'total_params' in checkpoint['state_dict']:
                checkpoint['state_dict'].pop('total_params')
                checkpoint['state_dict'].pop('total_ops')

            model.load_state_dict(checkpoint['state_dict'])
        else:
            logging.info("No checkpoint found at '{}', please check.".format(args.resume))
            return



    test_data = DND_sRGB_Dataset(path=args.root)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, shuffle=False, batch_size=1, num_workers=8, drop_last=False)

    model.eval()
    with torch.no_grad():
        for i, img_data in enumerate(tqdm(test_loader)):
            noisy_img = img_data[0].cuda()
            imgname   = img_data[1][0]
            clean_img = model(noisy_img)
            clean_img = torch.clamp(clean_img, 0, 1).cpu().numpy().squeeze().transpose(1, 2, 0)
            if args.save_img:
                save_img = img_as_ubyte(clean_img)
                utils.save_img(os.path.join(save_path, 'png/', imgname+ '.png'), save_img)
                mat_file_name = os.path.join(save_path, 'matfile/', imgname + '.mat')
                sio.savemat(mat_file_name, {'Idenoised_crop': np.float32(clean_img)})

    bundle_submission_srgb(os.path.join(save_path, 'matfile/'), 'srgb_results_for_server_submission/')
    os.system("rm {}".format(os.path.join(save_path, 'matfile/*.mat')))

if __name__ == '__main__':
    main()
