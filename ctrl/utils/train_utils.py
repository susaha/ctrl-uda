import torch
from pathlib import Path
import os
from torchvision.utils import make_grid
import torch.nn.functional as F
from ctrl.utils.viz_segmask import colorize_mask
import numpy as np


def print_output_paths(cfg, is_isl_training=None):
    print('*** output paths ***')
    print('cfg.TRAIN.SNAPSHOT_DIR: {}'.format(cfg.TRAIN.SNAPSHOT_DIR))
    print('cfg.TRAIN.SNAPSHOT_DIR_BESTMODEL: {}'.format(cfg.TRAIN.SNAPSHOT_DIR_BESTMODEL))
    print('cfg.TRAIN_LOG_FNAME: {}'.format(cfg.TRAIN_LOG_FNAME))
    print('cfg.TRAIN.TENSORBOARD_LOGDIR: {}'.format(cfg.TRAIN.TENSORBOARD_LOGDIR))
    print('cfg.TEST.VISUAL_RESULTS_DIR: {}'.format(cfg.TEST.VISUAL_RESULTS_DIR))
    if is_isl_training:
        print('cfg.TRAIN.PSEUDO_LABELS_DIR: {}'.format(cfg.TRAIN.PSEUDO_LABELS_DIR))
    print()


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** power)


def _adjust_learning_rate(optimizer, i_iter, cfg, learning_rate, DEBUG):
    lr = lr_poly(learning_rate, i_iter, cfg.TRAIN.MAX_ITERS, cfg.TRAIN.POWER)
    optimizer.param_groups[0]['lr'] = lr
    num_param_grps = len(optimizer.param_groups)
    if num_param_grps > 1:
        for i in range(1, num_param_grps):
            optimizer.param_groups[i]['lr'] = lr * 10


def adjust_learning_rate(optimizer, i_iter, cfg, DEBUG):
    _adjust_learning_rate(optimizer, i_iter, cfg, cfg.TRAIN.LEARNING_RATE, DEBUG=DEBUG)


def adjust_learning_rate_disc(optimizer, i_iter, cfg, DEBUG):
    _adjust_learning_rate(optimizer, i_iter, cfg, cfg.TRAIN.LEARNING_RATE_D, DEBUG=DEBUG)


def print_losses(current_losses, i_iter):
    list_strings = []
    for loss_name, loss_value in current_losses.items():
        list_strings.append(f'{loss_name} = {to_numpy(loss_value):.6f} ')
    full_string = ' '.join(list_strings)
    print(f'iter = {i_iter} {full_string}')


def to_numpy(tensor):
    if isinstance(tensor, (int, float)):
        return tensor
    else:
        return tensor.data.cpu().numpy()


def get_checkpoint_path(i_iter, cfg, current_epoch):
    snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
    bestmodel_dir = Path(cfg.TRAIN.SNAPSHOT_DIR_BESTMODEL)
    checkpoint_path = snapshot_dir / f"model_{i_iter}_{current_epoch}.pth"
    bestmodel_path = bestmodel_dir / f"model_{i_iter}_{current_epoch}.pth"
    checkpoint_path_tmp = snapshot_dir / f"model_{i_iter}_{current_epoch}.pth.tmp"
    return checkpoint_path, bestmodel_path, checkpoint_path_tmp


def save_checkpoint(i_iter, cfg, save_dict, checkpoint_path, checkpoint_path_tmp):
    snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
    if i_iter >= cfg.TRAIN.SAVE_PRED_EVERY * 2 and i_iter != 0:
        cp_list = [f for f in os.listdir(str(snapshot_dir)) if 'pth' in f]
        cp_list.sort(reverse=True)
        for f in cp_list:
            checkpoint_path_2_remove = os.path.join(str(snapshot_dir), f)
            strCmd2 = 'rm' + ' ' + checkpoint_path_2_remove
            print('Removing: {}'.format(strCmd2))
            os.system(strCmd2)
    print("Saving the checkpoint as tmp file at: {}".format(checkpoint_path_tmp))
    torch.save(save_dict, checkpoint_path_tmp)
    print("Moving the tmp checkpoint to actual checkpoint at: {}".format(checkpoint_path))
    strCmd = 'mv' + ' ' + str(checkpoint_path_tmp) + ' ' + str(checkpoint_path)
    os.system(strCmd)


def log_losses_tensorboard(writer, current_losses, i_iter):
    for loss_name, loss_value in current_losses.items():
        writer.add_scalar(f'data/{loss_name}', to_numpy(loss_value), i_iter)


def draw_in_tensorboard(writer, images, i_iter, pred_main, num_classes, type_):
    grid_image = make_grid(images[:3].clone().cpu().data, 3, normalize=True)
    writer.add_image(f"Image - {type_}", grid_image, i_iter)
    softmax = F.softmax(pred_main, dim=1).cpu().data[0].numpy().transpose(1, 2, 0)
    mask = colorize_mask(num_classes, np.asarray(np.argmax(softmax, axis=2), dtype=np.uint8)).convert("RGB")
    grid_image = make_grid(torch.from_numpy(np.array(mask).transpose(2, 0, 1)), 3, normalize=False, range=(0, 255))
    writer.add_image(f"Prediction - {type_}", grid_image, i_iter)
    output_sm = F.softmax(pred_main, dim=1).cpu().data[0].numpy().transpose(1, 2, 0)
    output_ent = np.sum(-np.multiply(output_sm, np.log2(output_sm + 1e-30)), axis=2, keepdims=False)
    grid_image = make_grid(torch.from_numpy(output_ent), 3, normalize=True, range=(0, np.log2(num_classes)))
    writer.add_image(f'Entropy - {type_}', grid_image, i_iter)


def per_class_iu(hist):
    np.seterr(divide='ignore', invalid='ignore')
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


