import torch
import numpy as np
import random
import sys
from ctrl.utils.logger import Logger
from ctrl.utils.common_config import get_model, get_criterion, get_optimizer,\
    get_optimizer_v2, get_data_loaders, convert_yaml_to_edict, setup_exp_params
from train import train_model
import yaml
import os
from ctrl.utils.train_utils import print_output_paths
import argparse


def main():
    DEBUG = False
    parser = argparse.ArgumentParser(description='CTRL-UDA Training')
    parser.add_argument('--machine', type=int, default=-1, help='which machine to use')
    parser.add_argument('--expid', type=int, default=1, help='experiment id')
    parser.add_argument('--reso', type=str, default='FULL', help='inputs resolution full or low')
    parser.add_argument('--isl', type=str, default='false', help='activate the ISL training')
    parser.add_argument('--exp_root_dir', type=str, help='experiment root folder')
    parser.add_argument('--data_root', type=str, help='dataset root folder')
    parser.add_argument('--pret_model', type=str, help='pretrained weights to be used for initialization')
    cmdline_inputs = parser.parse_args()
    expid = cmdline_inputs.expid
    if expid == 1:
        exp_file = 'ctrl/configs/synthia_to_cityscapes_16cls.yml'
    elif expid == 2:
        exp_file = 'ctrl/configs/synthia_to_cityscapes_7cls_fr.yml'
    elif expid == 3:
        exp_file = 'ctrl/configs/synthia_to_cityscapes_7cls_lr.yml'
    elif expid == 4:
        exp_file = 'ctrl/configs/synthia_to_mapillary_7cls_fr.yml'
    elif expid == 5:
        exp_file = 'ctrl/configs/synthia_to_mapillary_7cls_lr.yml'
    cfg = convert_yaml_to_edict(exp_file)
    cfg = setup_exp_params(cfg, cmdline_inputs, DEBUG)
    # set random seed
    torch.manual_seed(cfg.TRAIN.RANDOM_SEED)
    torch.cuda.manual_seed(cfg.TRAIN.RANDOM_SEED)
    np.random.seed(cfg.TRAIN.RANDOM_SEED)
    random.seed(cfg.TRAIN.RANDOM_SEED)
    # train, val logs
    sys.stdout = Logger(cfg.TRAIN_LOG_FNAME)
    # print output paths
    print_output_paths(cfg)
    # get model
    model, discriminator, optim_state_dict, disc_optim_state_dict, resume_iteration = get_model(cfg)
    if cfg.USE_DATA_PARALLEL:
        model = torch.nn.DataParallel(model)
    model = model.to(cfg.GPU_ID)
    if cfg.USE_DATA_PARALLEL:
        discriminator = torch.nn.DataParallel(discriminator)
    discriminator = discriminator.to(cfg.GPU_ID)
    # get criterion
    criterion_dict = get_criterion()
    if cfg.USE_DATA_PARALLEL:
        criterion_dict['semseg'] = torch.nn.DataParallel(criterion_dict['semseg'])
        criterion_dict['depth'] = torch.nn.DataParallel(criterion_dict['depth'])
        criterion_dict['disc_loss'] = torch.nn.DataParallel(criterion_dict['disc_loss'])
    criterion_dict['semseg'].to(cfg.GPU_ID)
    criterion_dict['depth'].to(cfg.GPU_ID)
    criterion_dict['disc_loss'].to(cfg.GPU_ID)
    print(criterion_dict)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    # get optimizer
    optimizer, optimizer_disc = get_optimizer_v2(cfg, model, USeDataParallel=cfg.USE_DATA_PARALLEL,
                                                  discriminator=discriminator, optim_state_dict=optim_state_dict,
                                                  disc_optim_state_dict=disc_optim_state_dict)
    print(optimizer)
    if cfg.ENABLE_DISCRIMINATOR:
        print(optimizer_disc)
    # dataloaders
    source_train_loader, target_train_loader, target_val_loader,\
    source_train_nsamp, target_train_nsamp, target_test_nsamp = get_data_loaders(cfg, get_target_train_loader=True)
    # dump cfg into a yml file
    cfg_file = os.path.join(cfg.TRAIN.SNAPSHOT_DIR, 'cfg.yml')
    with open(cfg_file, 'w') as fp:
        yaml.dump(dict(cfg), fp)
        print('cfg written to: {}'.format(cfg_file))
    # train the model
    train_model(cfg, model, discriminator, resume_iteration, criterion_dict, optimizer, optimizer_disc, source_train_loader,
                target_train_loader, target_val_loader, source_train_nsamp, target_train_nsamp, target_test_nsamp)


if __name__ == "__main__":
    main()




