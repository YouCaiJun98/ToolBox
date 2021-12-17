import os
import shutil
# Make Dir, Logging File and Save Scripts.
def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
             dst_file = os.path.join(path, 'scripts', os.path.basename(script))
             shutil.copyfile(script, dst_file)
        print('Scripts dir : {}'.format(os.path.join(path, 'scripts')))

import os
import shutil
import torch
# Pytorch model save
def save_checkpoint(state:dir, is_best:bool, save_dir:str, filename='checkpoint.pth.tar'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, filename)
    torch.save(state, save_path)
    if is_best:
        best_path = os.path.join(save_dir, 'model_best.pth.tar')
        shutil.copyfile(save_path, best_path)


# Since there are many methods to save/load params, we have to utilize a versatile 
# checkpoint loader to get them.
from collections import OrderedDict
def get_state_dict(checkpoint):
    if type(checkpoint) is OrderedDict:
        state_dict = checkpoint
    elif type(checkpoint) is dict:
        for key in checkpoint.keys():
            # In case the params of optimizer is saved in the checkpoint
            if ('state_dict' in key) and ('opti' not in key):
                state_dict = checkpoint[key]
    return state_dict

