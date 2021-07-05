import numpy as np
from torch.utils import data


def get_cityscape_train_dataloader(cfg):
    from ctrl.dataset.cityscapes import CityscapesDataSet
    def _init_fn(worker_id):
        print('WORKER ID : {} '.format(worker_id))
        np.random.seed(cfg.TRAIN.RANDOM_SEED + worker_id)
    if cfg.DEBUG:
        SHUFFLE = False
    else:
        SHUFFLE = True
    print('Creating target train dataset {}'.format(cfg.TARGET))
    print('cfg.TRAIN.SET_TARGET : {}'.format(cfg.TRAIN.SET_TARGET))
    print('cfg.TRAIN.MAX_ITERS : {}; cfg.TRAIN.BATCH_SIZE_TARGET: {}'.format(cfg.TRAIN.MAX_ITERS, cfg.TRAIN.BATCH_SIZE_TARGET))
    print('cfg.TRAIN.INPUT_SIZE_TARGET {}'.format(cfg.TRAIN.INPUT_SIZE_TARGET))
    print('SHUFFLE: {}'.format(SHUFFLE))
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
    target_train_dataset_num_examples = len(target_train_dataset)
    target_train_loader = data.DataLoader(
        target_train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_TARGET,
        num_workers=cfg.NUM_WORKERS,
        shuffle=SHUFFLE,
        pin_memory=True,
        worker_init_fn=_init_fn,
    )
    return target_train_loader, target_train_dataset_num_examples


def get_mapillary_train_dataloader(cfg):
    def _init_fn(worker_id):
        print('WORKER ID : {}'.format(worker_id))
        np.random.seed(cfg.TRAIN.RANDOM_SEED + worker_id)
    from ctrl.dataset.mapillary import MapillaryDataSet
    if cfg.DEBUG:
        SHUFFLE = False
    else:
        SHUFFLE = True
    print('Creating target train dataset {}'.format(cfg.TARGET))
    print('cfg.TRAIN.SET_TARGET : {}'.format(cfg.TRAIN.SET_TARGET))
    print('cfg.TRAIN.MAX_ITERS : {}; cfg.TRAIN.BATCH_SIZE_TARGET: {}'.format(cfg.TRAIN.MAX_ITERS, cfg.TRAIN.BATCH_SIZE_TARGET))
    print('cfg.TRAIN.INPUT_SIZE_TARGET {}'.format(cfg.TRAIN.INPUT_SIZE_TARGET))
    print('SHUFFLE: {}'.format(SHUFFLE))
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
    target_train_dataset_num_examples = len(target_train_dataset)
    target_train_loader = data.DataLoader(
        target_train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_TARGET,
        num_workers=cfg.NUM_WORKERS,
        shuffle=SHUFFLE,
        pin_memory=True,
        worker_init_fn=_init_fn,
    )
    return target_train_loader, target_train_dataset_num_examples



