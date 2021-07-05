import torch
from ctrl.utils.train_utils import per_class_iu, fast_hist
import numpy as np
from ctrl.utils.serialization import pickle_dump
from torch import nn
import os
import time

def mkdirs_ss(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print('folder created: {}'.format(path))


def display_stats(cfg, name_classes, inters_over_union_classes):
    for ind_class in range(cfg.NUM_CLASSES):
        print(name_classes[ind_class] + '\t' + str(round(inters_over_union_classes[ind_class] * 100, 2)))


def eval_model(model, test_loader, device, cfg, all_res, i_iter, cache_path, best_miou, checkpoint_path, best_model):
    DEBUG = cfg.DEBUG
    EXP_SETUP = cfg.EXP_SETUP
    print('class list:')
    print(test_loader.dataset.class_names)
    test_iter = iter(test_loader)
    str_target_dataset_name = EXP_SETUP.split('_')[2]
    if not cfg.TEST.OUTPUT_SIZE_TARGET:
        fixed_test_size = False
    else:
        fixed_test_size = True
    hist = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES))
    verbose = True
    num_val_samples = len(test_loader)
    print('number of val samples : {}'.format(num_val_samples))
    if DEBUG:
        num_val_samples = 10


    for index in range(num_val_samples):
        with torch.no_grad():
            if EXP_SETUP == 'SYNTHIA_TO_CITYSCAPES':
                image, label, _, _, img_name_target = next(test_iter)
            elif EXP_SETUP == 'SYNTHIA_TO_MAPILLARY':
                image, label, _, img_name_target = next(test_iter)
            if fixed_test_size:
                interp = nn.Upsample(size=(cfg.TEST.OUTPUT_SIZE_TARGET[1], cfg.TEST.OUTPUT_SIZE_TARGET[0]), mode='bilinear', align_corners=True)
            else:
                interp = nn.Upsample(size=(label.shape[1], label.shape[2]), mode='bilinear', align_corners=True)
            # forward pass
            # start_ts = time.time()
            pred_main = model(image.cuda(device))[1]
            # print('time taken: {}'.format( time.time() - start_ts))
            output = interp(pred_main).cpu().data[0].numpy()
            output = output.transpose(1, 2, 0)
            output = np.argmax(output, axis=2)
            label = label.numpy()[0]
            hist += fast_hist(label.flatten(), output.flatten(), cfg.NUM_CLASSES)
            if verbose and index > 0 and index % 100 == 0:
                print('{:d} / {:d}: {:0.2f}'.format(index, len(test_loader), 100 * np.nanmean(per_class_iu(hist))))
            if DEBUG:
                print('Evaluating Model on {} target dataset for SemSeg mIoU : {} ...'.format(str_target_dataset_name, index))
                if index % 10 == 0 and index != 0:
                    break
            else:
                if index % 50 == 0:
                    print('Evaluating Model on {} target dataset for SemSeg mIoU : {} ...'.format(str_target_dataset_name, index))
    inters_over_union_classes = per_class_iu(hist)
    all_res[i_iter] = inters_over_union_classes
    pickle_dump(all_res, cache_path)
    computed_miou = round(np.nanmean(inters_over_union_classes) * 100, 2)
    if best_miou < computed_miou:
        best_miou = computed_miou
        best_model = checkpoint_path
    print('\tCurrent mIoU:', computed_miou)
    print('\tCurrent model:', checkpoint_path)
    print('\tBest mIoU:', best_miou)
    print('\tBest model:', best_model)
    if verbose:
        display_stats(cfg, test_loader.dataset.class_names, inters_over_union_classes)
    return all_res, best_miou, best_model