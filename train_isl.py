from tensorboardX import SummaryWriter
import os
from torch import nn
import os.path as osp
from ctrl.utils.train_utils import adjust_learning_rate, adjust_learning_rate_disc, print_losses,\
    get_checkpoint_path, save_checkpoint, log_losses_tensorboard, draw_in_tensorboard
from eval import eval_model
import sys
from ctrl.isl.get_data_loader_for_isl import get_cityscape_train_dataloader, get_mapillary_train_dataloader

def train_model(cfg, model, discriminator, resume_iteration, criterion_dict, optimizer, optimizer_disc, source_train_loader,
                target_train_loader, target_val_loader, source_train_nsamp, target_train_nsamp, target_test_nsamp):

    target_train_loader = None
    loss_semseg = criterion_dict['semseg']
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    device = cfg.GPU_ID
    num_classes = cfg.NUM_CLASSES
    viz_tensorboard = os.path.exists(cfg.TRAIN.TENSORBOARD_LOGDIR)
    writer = None
    if viz_tensorboard:
        writer = SummaryWriter(log_dir=cfg.TRAIN.TENSORBOARD_LOGDIR)
    model.train()
    if cfg.ENABLE_DISCRIMINATOR:
        discriminator.train()
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode="bilinear", align_corners=True, )

    if resume_iteration:
        start_iter = resume_iteration + 1
    else:
        start_iter = 0
    all_res = {}
    all_res['semseg_pred'] = {}
    all_res['semseg_pred_given_depth'] = {}
    all_res['semseg_pred_fused'] = {}
    IS_BEST_ITER = False
    best_miou = -1
    best_model = ''
    cfg.TEST.SNAPSHOT_DIR = cfg.TRAIN.SNAPSHOT_DIR
    cache_path = osp.join(cfg.TEST.SNAPSHOT_DIR, 'all_res.pkl')
    local_iter = 0
    current_epoch = 0
    sudo_label_gen_iter = 0
    print()
    print('*** cfg ***')
    cfg_print = dict(cfg)
    for k, v in cfg_print.items():
        if isinstance(v, dict):
            for k1, v1 in v.items():
                print('cfg.{}.{}: {}'.format(k, k1, v1))
        else:
            print('cfg.{}: {}'.format(k, v))
    print()
    gen_sudo_labels = None

    # importing gen_sudo_labels for ISL training
    if cfg.EXP_SETUP == 'SYNTHIA_TO_CITYSCAPES':
        from ctrl.isl.gen_pseudo_labels_c import gen_sudo_labels
    elif cfg.EXP_SETUP == 'SYNTHIA_TO_MAPILLARY':
        from ctrl.isl.gen_pseudo_labels_m import gen_sudo_labels

    # generating pseudo labels
    print('generating pseudo labels at itration: {}'.format(start_iter))
    cfg.IS_ISL_TRAINING = None
    cfg.IS_ISL_TRAINING = gen_sudo_labels(model, cfg)
    if cfg.EXP_SETUP == 'SYNTHIA_TO_CITYSCAPES':
        target_train_loader, target_train_nsamp = get_cityscape_train_dataloader(cfg)
    elif cfg.EXP_SETUP == 'SYNTHIA_TO_MAPILLARY':
        target_train_loader, target_train_nsamp = get_mapillary_train_dataloader(cfg)
    target_train_loader_iter = enumerate(target_train_loader)

    epoch_iter = int(target_train_nsamp / cfg.TRAIN.BATCH_SIZE_SOURCE)
    count_epoch_in_source = False
    num_train_epochs = int(cfg.TRAIN.MAX_ITERS / epoch_iter)
    print('num iterations in one epoch: {}'.format(epoch_iter))
    print('total epoch to train: {}'.format(num_train_epochs))

    # train loop
    for i_iter in range(start_iter, cfg.TRAIN.EARLY_STOP + 1):
        local_iter += 1
        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter, cfg, cfg.DEBUG)
        # don't update disc while training generator
        for param in discriminator.parameters():
            param.requires_grad = False
        # generating pseudo labels
        if i_iter in cfg.GEN_PSEUDO_LABELS_EVERY:
            print('generating pseudo labels at itration: {}'.format(i_iter))
            sudo_label_gen_iter+=1
            cfg.IS_ISL_TRAINING = None
            cfg.IS_ISL_TRAINING = gen_sudo_labels(model, cfg)
            if cfg.EXP_SETUP == 'SYNTHIA_TO_CITYSCAPES':
                target_train_loader, target_train_dataset_num_examples = get_cityscape_train_dataloader(cfg)
            elif cfg.EXP_SETUP == 'SYNTHIA_TO_MAPILLARY':
                target_train_loader, target_train_dataset_num_examples = get_mapillary_train_dataloader(cfg)
            target_train_loader_iter = enumerate(target_train_loader)

        # supervised training on target
        try:
            _, batch = target_train_loader_iter.__next__()
        except StopIteration:
            target_train_loader_iter = enumerate(target_train_loader)
            _, batch = target_train_loader_iter.__next__()
            if not count_epoch_in_source:
                current_epoch += 1
        labels_target = None
        images_target = None
        if cfg.EXP_SETUP == 'SYNTHIA_TO_CITYSCAPES':
            images_target, labels_target, _, _, img_name_target = batch
        elif cfg.EXP_SETUP == 'SYNTHIA_TO_MAPILLARY':
            images_target, labels_target, _, img_name_target = batch
        labels_target = labels_target.to(device)
        _, semseg_pred_target, _, _ = model(images_target.to(device))
        semseg_pred_target = interp_target(semseg_pred_target)
        loss_sem_target = loss_semseg(semseg_pred_target, labels_target)
        loss = loss_sem_target
        if len(loss.size()) > 0:
            loss.mean().backward()
        else:
            loss.backward()
        optimizer.step()
        if len(loss_sem_target.size()) > 0:
            current_losses = {"loss_sem_target": loss_sem_target.mean(),}
        else:
            current_losses = {"loss_sem_target": loss_sem_target,}

        # print current losses
        if i_iter % cfg.TRAIN.DISPLAY_LOSS_RATE == cfg.TRAIN.DISPLAY_LOSS_RATE - 1 or i_iter == 0 or local_iter == 1:
            print_losses(current_losses, i_iter)

        # eval checkpoints
        checkpoint_path, bestmodel_path, checkpoint_path_tmp = get_checkpoint_path(i_iter, cfg, current_epoch)
        if i_iter % cfg.TRAIN.EVAL_EVERY == 0 and i_iter != 0 or i_iter == cfg.TRAIN.MAX_ITERS:
            model.eval()
            all_res, best_miou, best_model = eval_model(model, target_val_loader, device, cfg, all_res, i_iter,
                                                        cache_path, best_miou, checkpoint_path, best_model)
            str1 = str(checkpoint_path).split('/')[-1]
            str2 = str(best_model).split('/')[-1]
            if str1 == str2:
                IS_BEST_ITER = True
            else:
                IS_BEST_ITER = False
            model.train()

        # save checkpoint
        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0 or i_iter == cfg.TRAIN.MAX_ITERS or IS_BEST_ITER == True:
            if cfg.USE_DATA_PARALLEL:
                save_dict = {
                    'iter': i_iter,
                    'max_iter': cfg.TRAIN.MAX_ITERS,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }
            else:
                save_dict = {
                    'iter': i_iter,
                    'max_iter': cfg.TRAIN.MAX_ITERS,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }
            if IS_BEST_ITER == True and cfg.TRAIN.EVAL_EVERY != cfg.TRAIN.SAVE_PRED_EVERY:
                save_checkpoint(i_iter, cfg, save_dict, bestmodel_path, checkpoint_path_tmp)
                IS_BEST_ITER = False
            elif IS_BEST_ITER == True and cfg.TRAIN.EVAL_EVERY == cfg.TRAIN.SAVE_PRED_EVERY and i_iter == cfg.TRAIN.SAVE_PRED_EVERY:
                save_checkpoint(i_iter, cfg, save_dict, bestmodel_path, checkpoint_path_tmp)
                save_checkpoint(i_iter, cfg, save_dict, checkpoint_path, checkpoint_path_tmp)
                IS_BEST_ITER = False
            elif IS_BEST_ITER == True and cfg.TRAIN.EVAL_EVERY == cfg.TRAIN.SAVE_PRED_EVERY and i_iter != cfg.TRAIN.SAVE_PRED_EVERY:
                save_checkpoint(i_iter, cfg, save_dict, bestmodel_path, checkpoint_path_tmp)
                IS_BEST_ITER = False
            else:
                save_checkpoint(i_iter, cfg, save_dict, checkpoint_path, checkpoint_path_tmp)
        sys.stdout.flush()

        # tensorboard updates
        if viz_tensorboard:
            log_losses_tensorboard(writer, current_losses, i_iter)

        if i_iter % cfg.TRAIN.TENSORBOARD_VIZRATE == 0 and i_iter != 0 or i_iter == 100:
            print('Visualize in TensorBoard ... ')
            draw_in_tensorboard(writer, images_target, i_iter, semseg_pred_target, num_classes, "T")


















