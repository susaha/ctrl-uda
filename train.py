from ctrl.model.cross_task_relation import CTRL
from torch.utils.tensorboard import SummaryWriter
import os
from torch import nn
import os.path as osp
from ctrl.utils.train_utils import adjust_learning_rate, adjust_learning_rate_disc, print_losses,\
    get_checkpoint_path, save_checkpoint, log_losses_tensorboard, draw_in_tensorboard
from eval import eval_model
import sys


def train_model(cfg, model, discriminator, resume_iteration, criterion_dict, optimizer, optimizer_disc, source_train_loader,
                target_train_loader, target_val_loader, source_train_nsamp, target_train_nsamp, target_test_nsamp):

    ctrl = CTRL(cfg)
    loss_semseg = criterion_dict['semseg']
    loss_depth = criterion_dict['depth']
    bce_loss = criterion_dict['disc_loss']
    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    device = cfg.GPU_ID
    num_classes = cfg.NUM_CLASSES
    viz_tensorboard = os.path.exists(cfg.TRAIN.TENSORBOARD_LOGDIR)
    writer = None
    if viz_tensorboard:
        writer = SummaryWriter(log_dir=cfg.TRAIN.TENSORBOARD_LOGDIR)
    model.train()
    discriminator.train()
    interp = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode="bilinear", align_corners=True, )
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode="bilinear", align_corners=True, )
    source_label = 0
    target_label = 1
    source_train_loader_iter = enumerate(source_train_loader)
    target_train_loader_iter = enumerate(target_train_loader)
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
    count_epoch_in_source = True
    if source_train_nsamp > target_train_nsamp:
        epoch_iter = int(source_train_nsamp / cfg.TRAIN.BATCH_SIZE_SOURCE)
    else:
        epoch_iter = int(target_train_nsamp / cfg.TRAIN.BATCH_SIZE_SOURCE)
        count_epoch_in_source = False
    num_train_epochs = int(cfg.TRAIN.MAX_ITERS / epoch_iter)
    print('num iterations in one epoch: {}'.format(epoch_iter))
    print('total epoch to train: {}'.format(num_train_epochs))
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

    # train loop
    for i_iter in range(start_iter, cfg.TRAIN.EARLY_STOP + 1):
        local_iter += 1
        optimizer.zero_grad()
        optimizer_disc.zero_grad()
        adjust_learning_rate(optimizer, i_iter, cfg, cfg.DEBUG)
        adjust_learning_rate_disc(optimizer_disc, i_iter, cfg, cfg.DEBUG)

        # don't update disc while training generator
        for param in discriminator.parameters():
            param.requires_grad = False

        # supervised training on source
        try:
            _, batch = source_train_loader_iter.__next__()
        except StopIteration:
            source_train_loader_iter = enumerate(source_train_loader)
            _, batch = source_train_loader_iter.__next__()
            if count_epoch_in_source:
                current_epoch+=1
        images_source, labels, depth, _, img_name_source = batch
        labels = labels.to(device)
        depth = depth.to(device)
        _, semseg_pred_source, depth_pred_source, srh_pred_source = model(images_source.to(device))
        semseg_pred_source = interp(semseg_pred_source)
        depth_pred_source = interp(depth_pred_source)
        srh_pred_source = interp(srh_pred_source)
        loss_dep = loss_depth(depth_pred_source, depth)
        loss_sem = loss_semseg(semseg_pred_source, labels)
        loss_srh = loss_semseg(srh_pred_source, labels)
        loss = (cfg.TRAIN.LAMBDA_SEG * loss_sem + cfg.TRAIN.LAMBDA_SEG * loss_srh + cfg.TRAIN.LAMBDA_DEPTH * loss_dep)
        if len(loss.size()) > 0:
            loss.mean().backward()
        else:
            loss.backward()

        # adversarial training for generator
        try:
            _, batch = target_train_loader_iter.__next__()
        except StopIteration:
            target_train_loader_iter = enumerate(target_train_loader)
            _, batch = target_train_loader_iter.__next__()
            if not count_epoch_in_source:
                current_epoch+=1
        images_target = None
        if cfg.EXP_SETUP == 'SYNTHIA_TO_MAPILLARY':
            images_target, _, _, img_name_target = batch
        elif cfg.EXP_SETUP == 'SYNTHIA_TO_CITYSCAPES':
            images_target, _, _, _, img_name_target = batch
        _, semseg_pred_target, depth_pred_target, srh_pred_target = model(images_target.to(device))
        semseg_pred_target = interp_target(semseg_pred_target)
        srh_pred_target = interp_target(srh_pred_target)
        depth_pred_target = interp_target(depth_pred_target)

        # computing joint feature space for domain alignment
        Es = ctrl(semseg_pred_source, srh_pred_source, depth_pred_source)
        Et = ctrl(semseg_pred_target, srh_pred_target, depth_pred_target)
        d_out = discriminator(Et)
        loss_adv_trg = bce_loss(d_out, source_label)
        loss = cfg.TRAIN.LAMBDA_ADV_MAIN * loss_adv_trg
        if len(loss.size()) > 0:
            loss.mean().backward()
        else:
            loss.backward()

        # adversarial training for discriminator
        for param in discriminator.parameters():
            param.requires_grad = True
        Es = Es.detach()
        d_out = discriminator(Es)
        loss_d = bce_loss(d_out, source_label)
        loss_d = loss_d
        if len(loss_d.size()) > 0:
            loss_d.mean().backward()
        else:
            loss_d.backward()
        Et = Et.detach()
        d_out = discriminator(Et)
        loss_d = bce_loss(d_out, target_label)
        loss_d = loss_d
        if len(loss_d.size()) > 0:
            loss_d.mean().backward()
        else:
            loss_d.backward()

        # optimizers step
        optimizer.step()
        optimizer_disc.step()

        if len(loss_sem.size()) > 0:
            current_losses = {
                "loss_seg_src": loss_sem.mean(),
                "loss_srh_src": loss_srh.mean(),
                "loss_depth_src": loss_dep.mean(),
                "loss_adv_trg": loss_adv_trg.mean(),
                "loss_d": loss_d.mean(),
            }

        else:
            current_losses = {
                "loss_seg_src": loss_sem,
                "loss_srh_src": loss_srh,
                "loss_depth_src": loss_dep,
                "loss_adv_trg": loss_adv_trg,
                "loss_d": loss_d,
            }
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
                    'disc_state_dict': discriminator.module.state_dict(),
                    'disc_optim_state_dict': optimizer_disc.state_dict(),
                    'loss': loss,
                }
            else:
                save_dict = {
                    'iter': i_iter,
                    'max_iter': cfg.TRAIN.MAX_ITERS,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'disc_state_dict': discriminator.state_dict(),
                    'disc_optim_state_dict': optimizer_disc.state_dict(),
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
            draw_in_tensorboard(writer, images_source, i_iter, semseg_pred_source, num_classes, "S")

















