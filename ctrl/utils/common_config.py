from ctrl.model.mtl_aux_block import MTLAuxBlock
import torch
from ctrl.model.get_disc import get_discriminator
import numpy as np
from torch.utils import data
import yaml
from easydict import EasyDict as edict
from datetime import datetime
import os
import os.path as osp


def setup_exp_params(cfg, cmdline_inputs, DEBUG):
    cfg.GPU_ID = torch.device("cuda:0")
    cfg.DISC_INP_DIM = (cfg.NUM_CLASSES * 2) + cfg.NUM_DEPTH_BINS  # 16+16+15 = 47 ; or 7+7+15 = 29
    cfg.TRAIN.IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
    cfg.TEST.IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
    cfg.DEBUG = DEBUG
    cfg.MACHINE = cmdline_inputs.machine
    cfg.RESO = cmdline_inputs.reso
    cfg.IS_ISL = cmdline_inputs.isl
    if cfg.RESO == 'LOW':
       cfg.TRAIN.INPUT_SIZE_SOURCE = (640, 320)
       cfg.TRAIN.INPUT_SIZE_TARGET = (640, 320)
       cfg.TEST.INPUT_SIZE_TARGET = (640, 320)
       cfg.TEST.OUTPUT_SIZE_TARGET = (2048, 1024)
    elif cfg.RESO == 'FULL':
       cfg.TRAIN.INPUT_SIZE_SOURCE = (1280, 760)
       cfg.TRAIN.INPUT_SIZE_TARGET = (1024, 512)
       cfg.TEST.INPUT_SIZE_TARGET = (1024, 512)
       cfg.TEST.OUTPUT_SIZE_TARGET = (2048, 1024)
    if cfg.DEBUG:
        cfg.NUM_WORKERS = 0
        cfg.NUM_WORKERS_TEST = 0
        cfg.TRAIN.DISPLAY_LOSS_RATE = 1
        cfg.TRAIN.EVAL_EVERY = 10
        cfg.TRAIN.SAVE_PRED_EVERY = 10
        cfg.TRAIN.TENSORBOARD_VIZRATE = 10
        cfg.TRAIN.INPUT_SIZE_SOURCE = (640, 320)
        cfg.TRAIN.INPUT_SIZE_TARGET = (640, 320)
        cfg.TEST.INPUT_SIZE_TARGET = (640, 320)
        cfg.TEST.OUTPUT_SIZE_TARGET = (2048, 1024)
    if cfg.TARGET == 'Mapillary':
        cfg.TEST.OUTPUT_SIZE_TARGET = None
    if cfg.TARGET == 'Mapillary' and cfg.RESO == 'FULL':
        cfg.TRAIN.INPUT_SIZE_TARGET = (1024, 768)
        cfg.TEST.INPUT_SIZE_TARGET = (1024, 768)
    # experiment related params
    if cfg.TARGET == 'Mapillary':
        cfg.EXP_SETUP = 'SYNTHIA_TO_MAPILLARY'
    else:
        cfg.EXP_SETUP = 'SYNTHIA_TO_CITYSCAPES'
    exp_root = datetime.now().strftime("%m-%Y")
    phase_name = datetime.now().strftime("%d-%m-%Y")
    sub_phase_name = datetime.now().strftime("%H-%M-%S-%f")
    cfg.EXP_ROOT = 'exproot_{}'.format(exp_root)
    cfg.EXP_PHASE = 'phase_{}'.format(phase_name)
    cfg.EXP_SUB_PHASE = 'subphase_{}'.format(sub_phase_name)
    if cfg.MACHINE == 0:
        machine_spec_exp_root_dir = '/media/suman/DATADISK2/apps'
        cfg.DATA_ROOT = '/media/suman/DATADISK2/apps/datasets'
        pretrained_model_path = '/home/suman/apps/code/CVPR2021/MTI_Simon_ECCV2020/mti_simon/dada/pretrained_models'
    elif cfg.MACHINE == 1:
        machine_spec_exp_root_dir = '/raid/susaha'
        cfg.DATA_ROOT = '/raid/susaha/datasets'
        pretrained_model_path = '/raid/susaha/pretrained_models'
    elif cfg.MACHINE == 2:
        machine_spec_exp_root_dir = '/mnt/efs/fs1'
        cfg.DATA_ROOT = '/mnt/efs/fs1/datasets'
        pretrained_model_path = '/mnt/efs/fs1/cvpr_exp/pretrained_imagement'
    elif cfg.MACHINE == -1:
        machine_spec_exp_root_dir = cmdline_inputs.exp_root_dir
        cfg.DATA_ROOT = cmdline_inputs.data_root
        pretrained_model_path = cmdline_inputs.pret_model
    cfg.TRAIN.RESTORE_FROM = None
    os.makedirs('{}/{}'.format(machine_spec_exp_root_dir, cfg.EXP_ROOT), exist_ok=True)
    cfg.TRAIN.DADA_DEEPLAB_RESENT_PRETRAINED_IMAGENET = \
        '{}/DeepLab_resnet_pretrained_imagenet.pth'.format(pretrained_model_path)
    if cfg.TARGET == 'Mapillary':
        cfg.TRAIN.INFO_TARGET = 'ctrl/dataset/mapillary_list/info.json'.format(cfg.NUM_CLASSES)
        cfg.DATA_DIRECTORY_TARGET = osp.join(cfg.DATA_ROOT, 'Mapillary-Vista')
    else:
        cfg.TRAIN.INFO_TARGET = 'ctrl/dataset/cityscapes_list/info{}class.json'.format(cfg.NUM_CLASSES)
        cfg.DATA_DIRECTORY_TARGET = osp.join(cfg.DATA_ROOT, 'Cityscapes')
    cfg.TEST.INFO_TARGET = cfg.TRAIN.INFO_TARGET
    cfg.DATA_DIRECTORY_SOURCE = osp.join(cfg.DATA_ROOT, 'Synthia/RAND_CITYSCAPES')
    # if not cfg.WEIGHT_INITIALIZATION.RESUME_FROM_SNAPSHOT and not cfg.TRAIN.SNAPSHOT_DIR:
    cfg.TRAIN.SNAPSHOT_DIR = osp.join(machine_spec_exp_root_dir, cfg.EXP_ROOT, cfg.EXP_PHASE, 'checkpoints', cfg.EXP_SUB_PHASE)
    os.makedirs(cfg.TRAIN.SNAPSHOT_DIR, exist_ok=True)
    cfg.TRAIN_LOG_FNAME = os.path.join(cfg.TRAIN.SNAPSHOT_DIR, 'train_log.txt')
    cfg.TRAIN.SNAPSHOT_DIR_BESTMODEL = osp.join(cfg.TRAIN.SNAPSHOT_DIR, 'best_model')
    os.makedirs(cfg.TRAIN.SNAPSHOT_DIR_BESTMODEL, exist_ok=True)
    cfg.TRAIN.TENSORBOARD_LOGDIR = cfg.TRAIN.SNAPSHOT_DIR.replace('checkpoints', 'tensorboard')
    os.makedirs(cfg.TRAIN.TENSORBOARD_LOGDIR, exist_ok=True)
    cfg.TEST.VISUAL_RESULTS_DIR = cfg.TRAIN.SNAPSHOT_DIR.replace('checkpoints', 'visual_results')
    os.makedirs(cfg.TEST.VISUAL_RESULTS_DIR, exist_ok=True)
    if cfg.IS_ISL == 'true':
        cfg.WEIGHT_INITIALIZATION.RESUME_FROM_SNAPSHOT_GIVEN = cmdline_inputs.model_path
        cfg.TRAIN.PSEUDO_LABELS_DIR = cfg.TRAIN.SNAPSHOT_DIR.replace('checkpoints', 'pseudo_labels')
        os.makedirs(cfg.TRAIN.PSEUDO_LABELS_DIR, exist_ok=True)
        cfg.PSEUDO_LABELS_SUBDIR = 'labels_{}-{}'.format(phase_name, sub_phase_name)
        if cfg.DEBUG:
            cfg.GEN_PSEUDO_LABELS_EVERY = [11, 21, 31, 41, 51, 61, 71, 81, 91]
        else:
            cfg.GEN_PSEUDO_LABELS_EVERY = [10001, 20001, 30001, 40001, 50001, 60001, 70001, 80001, 90000]
    return cfg


def convert_yaml_to_edict(exp_file):
    with open(exp_file, 'r') as stream:
        config = yaml.safe_load(stream)
    cfg = edict()
    for k, v in config.items():
        if type(v) is dict:
            v = edict(v)
        cfg[k] = v
    return cfg


def get_model(cfg=None):
    model = MTLAuxBlock(cfg.NUM_CLASSES)
    checkpoint = None
    discriminator = None
    optim_state_dict = None
    disc_optim_state_dict = None
    resume_iteration = None
    if cfg.WEIGHT_INITIALIZATION.DADA_DEEPLABV2:
        model_params_current = model.state_dict().copy()
        model_params_saved = torch.load(cfg.TRAIN.DADA_DEEPLAB_RESENT_PRETRAINED_IMAGENET)
        for i in model_params_saved:
            i_parts = i.split(".")
            if not i_parts[1] == "layer5":
                model_params_current['backbone.{}'.format(".".join(i_parts[1:]))] = model_params_saved[i]
        model.load_state_dict(model_params_current)

    elif cfg.WEIGHT_INITIALIZATION.RESUME_FROM_SNAPSHOT:
        if not cfg.WEIGHT_INITIALIZATION.RESUME_FROM_SNAPSHOT_GIVEN:
            cpfiles = os.listdir(cfg.TRAIN.SNAPSHOT_DIR)
            cpfiles = [f for f in cpfiles if '.pth' in f]
            cpfiles.sort(reverse=True)
            cfg.WEIGHT_INITIALIZATION.RESUME_FROM_SNAPSHOT_GIVEN = os.path.join(cfg.TRAIN.SNAPSHOT_DIR, cpfiles[0])
        print('Resuming from checkpoint: {}'.format(cfg.WEIGHT_INITIALIZATION.RESUME_FROM_SNAPSHOT_GIVEN))
        checkpoint = torch.load(cfg.WEIGHT_INITIALIZATION.RESUME_FROM_SNAPSHOT_GIVEN)
        model.load_state_dict(checkpoint['model_state_dict'])
        optim_state_dict = checkpoint['optimizer_state_dict']
        if not cfg.TRAIN_ISL_FROM_SCRATCH:
            resume_iteration = checkpoint['iter']

    if cfg.ENABLE_DISCRIMINATOR:
        discriminator = get_discriminator(num_classes=cfg.DISC_INP_DIM)
        if cfg.WEIGHT_INITIALIZATION.RESUME_FROM_SNAPSHOT:
            discriminator.load_state_dict(checkpoint['disc_state_dict'])
            disc_optim_state_dict = checkpoint['disc_optim_state_dict']

    return model, discriminator, optim_state_dict, disc_optim_state_dict, resume_iteration


def get_criterion():
    criterion_dict = {}
    from ctrl.utils.loss_functions import CrossEntropy2D
    criterion_dict['semseg'] = CrossEntropy2D()
    from ctrl.utils.loss_functions import BerHuLossDepth
    criterion_dict['depth'] = BerHuLossDepth()
    from ctrl.utils.loss_functions import BCELossSS
    criterion_dict['disc_loss'] = BCELossSS()
    return criterion_dict


def get_optimizer(cfg, model, USeDataParallel=None, discriminator=None, optim_state_dict=None, disc_optim_state_dict=None):
    optimizer_discriminator = None
    if USeDataParallel:
        optim_list_backbone = model.module.backbone.optim_parameters(cfg.TRAIN.LEARNING_RATE)
        optim_list_encoder = model.module.encoder.optim_parameters(cfg.TRAIN.LEARNING_RATE)
        optim_list_decoder_single_conv = model.module.decoder_single_conv.optim_parameters(cfg.TRAIN.LEARNING_RATE)
        optim_list_decoder_semseg = model.module.decoder_semseg.optim_parameters(cfg.TRAIN.LEARNING_RATE)
        optim_list_depth_head = model.module.depth_head.optim_parameters(cfg.TRAIN.LEARNING_RATE)
        optim_list_decoder_semseg_given_depth = model.module.decoder_semseg_given_depth.optim_parameters(cfg.TRAIN.LEARNING_RATE)
    else:
        optim_list_backbone = model.backbone.optim_parameters(cfg.TRAIN.LEARNING_RATE)
        optim_list_encoder = model.encoder.optim_parameters(cfg.TRAIN.LEARNING_RATE)
        optim_list_decoder_single_conv = model.decoder_single_conv.optim_parameters(cfg.TRAIN.LEARNING_RATE)
        optim_list_decoder_semseg = model.decoder_semseg.optim_parameters(cfg.TRAIN.LEARNING_RATE)
        optim_list_depth_head = model.module.depth_head.optim_parameters(cfg.TRAIN.LEARNING_RATE)
        optim_list_decoder_semseg_given_depth = model.decoder_semseg_given_depth.optim_parameters(cfg.TRAIN.LEARNING_RATE)
    optim_list = optim_list_backbone + optim_list_encoder + optim_list_decoder_single_conv + \
                 optim_list_decoder_semseg + optim_list_depth_head + optim_list_decoder_semseg_given_depth
    optimizer = torch.optim.SGD(optim_list, lr=cfg.TRAIN.LEARNING_RATE, momentum=cfg.TRAIN.MOMENTUM, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    if cfg.ENABLE_DISCRIMINATOR:
        optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D, betas=(0.9, 0.99))
    if not cfg.TRAIN_ISL_FROM_SCRATCH:
        if optim_state_dict and cfg.WEIGHT_INITIALIZATION.RESUME_FROM_SNAPSHOT:
            optimizer.load_state_dict(optim_state_dict)
            print('model optimizer is loaded from: {}'.format(cfg.WEIGHT_INITIALIZATION.RESUME_FROM_SNAPSHOT_GIVEN))
        if cfg.ENABLE_DISCRIMINATOR:
            if disc_optim_state_dict and cfg.WEIGHT_INITIALIZATION.RESUME_FROM_SNAPSHOT:
                optimizer_discriminator.load_state_dict(disc_optim_state_dict)
                print('discriminator optimizer is loaded from: {}'.format(cfg.WEIGHT_INITIALIZATION.RESUME_FROM_SNAPSHOT_GIVEN))
    return optimizer, optimizer_discriminator


def get_optimizer_v2(cfg, model, USeDataParallel=None, discriminator=None, optim_state_dict=None, disc_optim_state_dict=None):
    optimizer_discriminator = None
    if USeDataParallel:
        optim_list_backbone = model.module.backbone.optim_parameters(cfg.TRAIN.LEARNING_RATE)
        optim_list_decoder = model.module.decoder.optim_parameters(cfg.TRAIN.LEARNING_RATE)
        optim_list_head_seg = model.module.head_seg.optim_parameters(cfg.TRAIN.LEARNING_RATE)
        optim_list_head_dep = model.module.head_dep.optim_parameters(cfg.TRAIN.LEARNING_RATE)
        optim_list_head_srh = model.module.head_srh.optim_parameters(cfg.TRAIN.LEARNING_RATE)
    else:
        optim_list_backbone = model.backbone.optim_parameters(cfg.TRAIN.LEARNING_RATE)
        optim_list_decoder = model.decoder.optim_parameters(cfg.TRAIN.LEARNING_RATE)
        optim_list_head_seg = model.head_seg.optim_parameters(cfg.TRAIN.LEARNING_RATE)
        optim_list_head_dep = model.head_dep.optim_parameters(cfg.TRAIN.LEARNING_RATE)
        optim_list_head_srh = model.head_srh.optim_parameters(cfg.TRAIN.LEARNING_RATE)
    optim_list = optim_list_backbone + optim_list_decoder + optim_list_head_seg + optim_list_head_dep + optim_list_head_srh
    optimizer = torch.optim.SGD(optim_list, lr=cfg.TRAIN.LEARNING_RATE, momentum=cfg.TRAIN.MOMENTUM, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    if cfg.ENABLE_DISCRIMINATOR:
        optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D, betas=(0.9, 0.99))
    if not cfg.TRAIN_ISL_FROM_SCRATCH:
        if optim_state_dict and cfg.WEIGHT_INITIALIZATION.RESUME_FROM_SNAPSHOT:
            optimizer.load_state_dict(optim_state_dict)
            print('model optimizer is loaded from: {}'.format(cfg.WEIGHT_INITIALIZATION.RESUME_FROM_SNAPSHOT_GIVEN))
        if cfg.ENABLE_DISCRIMINATOR:
            if disc_optim_state_dict and cfg.WEIGHT_INITIALIZATION.RESUME_FROM_SNAPSHOT:
                optimizer_discriminator.load_state_dict(disc_optim_state_dict)
                print('discriminator optimizer is loaded from: {}'.format(cfg.WEIGHT_INITIALIZATION.RESUME_FROM_SNAPSHOT_GIVEN))
    return optimizer, optimizer_discriminator


def get_data_loaders(cfg, get_target_train_loader=True):
    def _init_fn(worker_id):
        print('WORKER ID : {} '.format(worker_id))
        np.random.seed(cfg.TRAIN.RANDOM_SEED + worker_id)
    source_train_dataset = None
    target_train_dataset = None
    target_test_dataset = None
    source_train_loader = None
    target_train_loader = None
    source_train_nsamp = None
    target_train_nsamp = None
    if cfg.SOURCE == 'SYNTHIA':
        from ctrl.dataset.synthia import SYNTHIADataSetDepth
        source_train_dataset = SYNTHIADataSetDepth(
            root=cfg.DATA_DIRECTORY_SOURCE,
            list_path=cfg.DATA_LIST_SOURCE,
            set=cfg.TRAIN.SET_SOURCE,
            num_classes=cfg.NUM_CLASSES,
            max_iters=None,
            crop_size=cfg.TRAIN.INPUT_SIZE_SOURCE,
            mean=cfg.TRAIN.IMG_MEAN,
            use_depth=cfg.USE_DEPTH,
            depth_processing=cfg.DEPTH_PROCESSING,
            cfg=cfg,
            joint_transform=None,
        )
    if cfg.SOURCE:
        source_train_loader = data.DataLoader(
            source_train_dataset,
            batch_size=cfg.TRAIN.BATCH_SIZE_SOURCE,
            num_workers=cfg.NUM_WORKERS,
            shuffle=True,
            pin_memory=True,
            worker_init_fn=_init_fn,
        )
    if cfg.TARGET == 'Mapillary':
        from ctrl.dataset.mapillary import MapillaryDataSet
        if get_target_train_loader:
            target_train_dataset = MapillaryDataSet(
            root=cfg.DATA_DIRECTORY_TARGET,
            list_path=cfg.DATA_LIST_TARGET,
            set=cfg.TRAIN.SET_TARGET,
            info_path=cfg.TRAIN.INFO_TARGET,
            max_iters=None,
            crop_size=cfg.TRAIN.INPUT_SIZE_TARGET,
            mean=cfg.TRAIN.IMG_MEAN,
            scale_label=True,
            joint_transform=None,
            cfg=cfg,
        )
        target_test_dataset = MapillaryDataSet(
            root=cfg.DATA_DIRECTORY_TARGET,
            list_path=cfg.DATA_LIST_TARGET,
            set=cfg.TEST.SET_TARGET,
            info_path=cfg.TEST.INFO_TARGET,
            crop_size=cfg.TEST.INPUT_SIZE_TARGET,
            mean=cfg.TRAIN.IMG_MEAN,
            scale_label=False,
            joint_transform=None,
            cfg=cfg,
        )
    elif cfg.TARGET == 'Cityscapes':
        from ctrl.dataset.cityscapes import CityscapesDataSet
        if get_target_train_loader:
            target_train_dataset = CityscapesDataSet(
                root=cfg.DATA_DIRECTORY_TARGET,
                list_path=cfg.DATA_LIST_TARGET,
                set=cfg.TRAIN.SET_TARGET,
                info_path=cfg.TRAIN.INFO_TARGET,
                max_iters=None,
                crop_size=cfg.TRAIN.INPUT_SIZE_TARGET,
                mean=cfg.TRAIN.IMG_MEAN,
                joint_transform=None,
                cfg=cfg,
            )
        target_test_dataset = CityscapesDataSet(
            root=cfg.DATA_DIRECTORY_TARGET,
            list_path=cfg.DATA_LIST_TARGET,
            set=cfg.TEST.SET_TARGET,
            info_path=cfg.TEST.INFO_TARGET,
            crop_size=cfg.TEST.INPUT_SIZE_TARGET,
            mean=cfg.TEST.IMG_MEAN,
            labels_size=cfg.TEST.OUTPUT_SIZE_TARGET,
            joint_transform=None,
            cfg=cfg,
        )

    if get_target_train_loader:
        target_train_loader = data.DataLoader(
            target_train_dataset,
            batch_size=cfg.TRAIN.BATCH_SIZE_TARGET,
            num_workers=cfg.NUM_WORKERS,
            shuffle=True,
            pin_memory=True,
            worker_init_fn=_init_fn,
        )

    target_val_loader = data.DataLoader(
        target_test_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_TARGET,
        num_workers=cfg.NUM_WORKERS_TEST,
        shuffle=False,
        pin_memory=True,
        worker_init_fn=_init_fn,
    )
    if cfg.SOURCE:
        source_train_nsamp = len(source_train_dataset)
        print('{} : source train examples: {}'.format(cfg.SOURCE, source_train_nsamp))
    if target_train_dataset:
        target_train_nsamp = len(target_train_dataset)
    target_test_nsamp = len(target_test_dataset)
    if target_train_dataset:
        print('{} : target train examples: {}'.format(cfg.TARGET, target_train_nsamp))
    print('{} : target test examples: {}'.format(cfg.TARGET, target_test_nsamp))
        
    return source_train_loader, target_train_loader, target_val_loader, source_train_nsamp, target_train_nsamp, target_test_nsamp








