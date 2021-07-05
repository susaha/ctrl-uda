from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils import data
import json
import cv2


class BaseDataset(data.Dataset):
    def __init__(self, root, list_path, set_, max_iters, image_size, labels_size, mean, joint_transform, cfg):
        if 'Synthia' in root:
            self.dataset_name = 'Synthia'
        elif 'GTA' in root:
            self.dataset_name = 'GTA'
        if 'Mapillary' in root:
            self.dataset_name = 'Mapillary'
        elif 'Cityscapes' in root:
            self.dataset_name = 'Cityscapes'
        elif 'VKITTI' in root:
            self.dataset_name = 'VKITTI'
        elif 'KITTI' in root:
            self.dataset_name = 'KITTI'
        self.cfg = cfg
        self.root = Path(root)
        self.set = set_
        self.list_path = list_path.format(self.set)
        self.image_size = image_size
        self.joint_transform = joint_transform
        if labels_size is None:
            self.labels_size = self.image_size
        else:
            self.labels_size = labels_size
        self.mean = mean
        with open(self.list_path) as f:
            self.img_ids = [i_id.strip() for i_id in f]
        if self.cfg.DEBUG and cfg.NUM_TRAIN_SAMPLES_IN_SOURCE_TARGET:
            print('*** Under debug mode selecting {} samples for dataset {} ***'.format(cfg.NUM_TRAIN_SAMPLES_IN_SOURCE_TARGET, self.dataset_name))
            self.img_ids = self.img_ids[:cfg.NUM_TRAIN_SAMPLES_IN_SOURCE_TARGET]
        if max_iters is not None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        print('ctrl/dataset/base_dataset.py --> class BaseDataset --> __init__()')
        if 'Synthia' in root:
            print('cfg.SYNTHIA_DATALOADING_MODE: {}'.format(self.cfg.SYNTHIA_DATALOADING_MODE))
            mode = self.cfg.SYNTHIA_DATALOADING_MODE
            show_sample_files = True
            for name in self.img_ids:
                if mode == 'original_only' or mode == 'translated_only':
                    img_file, label_file = self.get_metadata(name, mode=mode)
                    self.files.append((img_file, label_file, name))
                    if show_sample_files:
                        print('Sample image locations:')
                        print(img_file)
                        show_sample_files =False
                else:
                    img_file1, img_file2, label_file = self.get_metadata(name, mode=mode)
                    self.files.append((img_file1, label_file, name))
                    self.files.append((img_file2, label_file, name))
                    if show_sample_files:
                        print('Sample image locations:')
                        print(img_file1)
                        print(img_file2)
                        show_sample_files =False
            print('creating image filename list for Synthia {} set ...'.format(self.set))
        elif 'VKITTI' in root:
            show_sample_files = True
            for name in self.img_ids:
                img_file, label_file = self.get_metadata(name, mode=None)
                self.files.append((img_file, label_file, name))
                if show_sample_files:
                    print('Sample image locations:')
                    print(img_file)
                    show_sample_files = False
            print('creating image filename list for {} {} set ...'.format(self.dataset_name, self.set))
        elif 'KITTI' in root:
            show_sample_files = True
            for name in self.img_ids:
                if self.set == 'train':
                    img_file, label_file = self.get_metadata(name, mode=None)
                    self.files.append((img_file, label_file, name))
                elif self.set == 'test' and self.cfg.IS_ISL_TRAINING:
                    img_file, label_file = self.get_metadata(name, mode=None)
                    self.files.append((img_file, label_file, name))
                elif self.set == 'test' and not self.cfg.IS_ISL_TRAINING:
                    img_file = self.get_metadata(name, mode=None)
                    self.files.append((img_file, name))
                if show_sample_files:
                    print('Sample image locations:')
                    print(img_file)
                    show_sample_files = False
            print('creating image filename list for {} {} set ...'.format(self.dataset_name, self.set))
        elif 'GTA' in root or 'Mapillary' in root:
            for name in self.img_ids:
                img_file, label_file = self.get_metadata(name)
                self.files.append((img_file, label_file, name))
            if 'GTA' in root:
                print('creating image filename list for GTA {} set ...'.format(self.set))
            if 'Mapillary' in root:
                print('creating image filename list for Mapillary {} set ...'.format(self.set))
        elif 'Cityscapes' in root:
            for name in self.img_ids:
                img_file, label_file, disp_file, calib_file = self.get_metadata(name)
                self.files.append((img_file, label_file, disp_file, calib_file, name))
            print('creating image filename list for Cityscapes {} set...'.format(self.set))
        else:
            raise NotImplementedError('entry not present for this dataset, make an entry here!!')

    def get_metadata(self, name, mode=None):
        raise NotImplementedError

    def __len__(self):
        return len(self.files)

    def preprocess(self, image):
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        return image.transpose((2, 0, 1))

    def get_image(self, file):
        return _load_img(file, self.image_size, Image.BICUBIC, rgb=True)

    def get_labels(self, file):
        return _load_img(file, self.labels_size, Image.NEAREST, rgb=False)

    def apply_data_augment(self, inputs):
        img, semseg_gt, depth_gt = self.joint_transform(inputs)
        return img, semseg_gt, depth_gt

    def get_depth_labels(self, disp_img_file, calib_json_file):
        with open(calib_json_file) as json_file:
            calib = json.load(json_file)
        baseline = calib['extrinsic']['baseline']
        focal_length = calib['intrinsic']['fx']
        DISPARITY_INVALID_VALUE = 0.0
        img = _load_img(disp_img_file, self.labels_size, Image.NEAREST, rgb=False)
        assert len(img.shape) == 2, 'Image {} shape is not 2D, but {}'.format(disp_img_file, img.shape)
        mask_invalid = img <= DISPARITY_INVALID_VALUE
        img[mask_invalid] = np.nan
        img = img - 1
        with np.errstate(invalid='ignore'):
            mask_disparity_0 = img <= 0.0
            mask_disparity_gt_0 = img > 0.0
        img = img / 256
        disparity_min = np.amin(img[mask_disparity_gt_0])
        depth_max = (baseline * focal_length) / disparity_min
        with np.errstate(divide='ignore'):
            img = (baseline * focal_length) / img  # computing the depth
        img[mask_disparity_0] = depth_max
        if self.set == 'val':
            return img
        elif self.set == 'train':
            img = cv2.resize(img, tuple(self.labels_size), interpolation=cv2.INTER_NEAREST)
            mask_valid = img == img
            img[mask_valid] = img[mask_valid] * 100
            img[mask_valid] = 65536.0 / (img[mask_valid] + 1)  # inverse depth
            return img


def _load_img(file, size, interpolation, rgb):
    img = Image.open(file)
    if rgb:
        img = img.convert('RGB')
    img = img.resize(size, interpolation)
    return np.asarray(img, np.float32)
