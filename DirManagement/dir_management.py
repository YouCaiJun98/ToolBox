import os
import shutil
# Make Dir and Save Scripts.
def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
             dst_file = os.path.join(path, 'scripts', os.path.basename(script))
             shutil.copyfile(script, dst_file)

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

