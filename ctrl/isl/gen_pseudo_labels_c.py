import torch.nn
import torch.nn as nn
from PIL import Image
import os
import numpy as np
import torch
import torch.nn.functional as F
from ctrl.isl.get_data_loader_for_isl import get_cityscape_train_dataloader


def gen_sudo_labels(model, cfg):
    model.eval()
    print('ctrl/isl/gen_pseudo_labels_c.py --> gen_sudo_labels(...)')
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    device = cfg.GPU_ID
    num_classes = cfg.NUM_CLASSES
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode="bilinear", align_corners=True, )
    target_train_loader, target_train_dataset_num_examples = get_cityscape_train_dataloader(cfg)
    target_train_loader_iter = enumerate(target_train_loader)
    save_root = os.path.join(cfg.TRAIN.PSEUDO_LABELS_DIR, 'cityscapes', 'train') # arxiv version
    save_path = os.path.join(save_root, cfg.PSEUDO_LABELS_SUBDIR, 'nparrays_{:.1f}'.format(cfg.ISL_THRESHOLD))
    save_path_imgs = os.path.join(save_root, cfg.PSEUDO_LABELS_SUBDIR, 'imgs_{:.1f}'.format(cfg.ISL_THRESHOLD))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print('folder created: {}'.format(save_path))
    if not os.path.exists(save_path_imgs):
        os.makedirs(save_path_imgs)
        print('folder created: {}'.format(save_path_imgs))
    for i_iter in range(0, target_train_dataset_num_examples+1):
        with torch.no_grad():
            try:
                _, batch = target_train_loader_iter.__next__()
            except StopIteration:
                target_train_loader_iter = enumerate(target_train_loader)
                _, batch = target_train_loader_iter.__next__()

            if cfg.EXP_SETUP == 'SYNTHIA_TO_MAPILLARY':
                images_target, labels, _, img_name_target = batch
            else:
                images_target, labels, _, _, img_name_target = batch
            # generating folders and filenames
            str = img_name_target[0].split('/')
            save_path_nparray = os.path.join(save_path, str[0])
            save_path_images = os.path.join(save_path_imgs, str[0])
            if not os.path.exists(save_path_nparray):
                os.makedirs(save_path_nparray)
                os.makedirs(save_path_images)
                if cfg.DEBUG:
                    print('creatining dir: {}'.format(save_path_nparray))
                    print('creatining dir: {}'.format(save_path_images))
                elif i_iter % 200 == 0 or i_iter == 0:
                    print('creatining dir: {}'.format(save_path_nparray))
                    print('creatining dir: {}'.format(save_path_images))
            _, semseg_pred_target, _, _ = model(images_target.to(device))
            semseg_pred_target = interp_target(semseg_pred_target)
            _, numc, pred_h, pred_w = semseg_pred_target.size()
            assert numc == num_classes, 'error'
            # converting the semseg predicted logits to softmax scores using the threshold
            semseg_pred_target = F.softmax(semseg_pred_target, dim=1)
            # masking out the scores below the threshold
            semseg_pred_target[semseg_pred_target >= cfg.ISL_THRESHOLD] = 1
            semseg_pred_target[semseg_pred_target < cfg.ISL_THRESHOLD] = -1
            # here we store the final sudo label, fill the numpy array with 255 which is the background label id
            sudo_labels = torch.zeros(pred_h, pred_w).long()
            torch.fill_(sudo_labels, 255)
            sudo_labels_vis = torch.zeros(pred_h, pred_w).long()
            torch.fill_(sudo_labels_vis, 0)
            # updating the numpy array with the respective class ids 0,1,2..., 15
            for i in range(numc):
                mask1 = semseg_pred_target[:, i, :, :] == 1
                sudo_labels[mask1.squeeze()] = i
                sudo_labels_vis[mask1.squeeze()] = (i + 1) * 10
            sudo_labels = sudo_labels.detach().cpu().numpy()
            str2 = str[1].split('.')
            str2 = '{}.npy'.format(str2[0])
            filename = os.path.join(save_path_nparray, str2)
            if cfg.DEBUG:
                print('writing to file: {}'.format(filename))
            elif i_iter % 200 == 0 or i_iter == 0:
                print('writing to file: {}'.format(filename))
            with open(filename, 'wb') as f:
                np.save(f, sudo_labels)
            filename2 = os.path.join(save_path_images, str[1])
            im2disp = sudo_labels_vis.squeeze().clone().detach().cpu().numpy()
            im2disp = Image.fromarray(im2disp.astype('uint8'), 'L')
            im2disp.save(filename2)
            if cfg.DEBUG:
                print('writing to file: {}'.format(filename2))
                print('>>> num images done: {}'.format(i_iter))
            elif i_iter % 200 == 0 or i_iter == 0:
                print('writing to file: {}'.format(filename2))
                print('>>> num images done: {}'.format(i_iter))
    model.train()
    return True










