import numpy as np
from ctrl.utils.serialization import json_load
from ctrl.dataset.base_dataset import BaseDataset
import os
import cv2


class CityscapesDataSet(BaseDataset):
    def __init__(self, root, list_path, set='val', max_iters=None, crop_size=(321, 321),
                 mean=(128, 128, 128), load_labels=True, info_path=None, labels_size=None,
                 transform=None, joint_transform=None, cfg=None):
        super().__init__(root, list_path, set, max_iters, crop_size, labels_size, mean, joint_transform, cfg)

        print('ctrl/dataset/cityscapes.py --> class CityscapesDataSet --> __init__() +++')
        print('self.cfg.IS_ISL_TRAINING: {}'.format(self.cfg.IS_ISL_TRAINING))

        self.load_labels = load_labels
        self.info = json_load(info_path)
        self.class_names = np.array(self.info['label'], dtype=np.str)
        self.mapping = np.array(self.info['label2train'], dtype=np.int)
        self.map_vector = np.zeros((self.mapping.shape[0],), dtype=np.int64)
        self.joint_transform = joint_transform
        for source_label, target_label in self.mapping:
            self.map_vector[source_label] = target_label

    def get_metadata(self, name, mode=None):
        img_file = self.root / 'leftImg8bit_trainvaltest/leftImg8bit' / self.set / name
        if self.cfg.IS_ISL_TRAINING and self.set == 'train':
            str1 = name.split('/')
            str2 = str1[1].split('.')
            str3 = os.path.join(str1[0], str2[0] + '.npy')
            # arxiv version
            label_file = os.path.join(self.cfg.TRAIN.PSEUDO_LABELS_DIR, 'cityscapes', 'train', self.cfg.PSEUDO_LABELS_SUBDIR,
                                      'nparrays_{:.1f}'.format(self.cfg.ISL_THRESHOLD), str3)
            # cvpr submitted version
            # label_file = os.path.join(self.root, 'gtFinePseudo_trainvaltest/gtFine',
            #                          self.set, self.cfg.MODEL_FOR_SUDO_LABEL_TRAINING,
            #                          'nparrays_{:.1f}'.format(self.cfg.ISL_THRESHOLD), str3)
        else:
            label_name = name.replace("leftImg8bit", "gtFine_labelIds")
            label_file = self.root / 'gtFine_trainvaltest/gtFine' / self.set / label_name

        label_name = name.replace("leftImg8bit", "disparity")
        disp_img_file = self.root / 'disparity_trainvaltest/disparity' / self.set / label_name
        label_name = name.replace("leftImg8bit", "camera")
        label_name = label_name.replace("png", "json")
        disp_json_file = self.root / 'camera_trainvaltest/camera' / self.set / label_name
        return img_file, label_file, disp_img_file, disp_json_file

    def map_labels(self, input_):
        return self.map_vector[input_.astype(np.int64, copy=False)]

    def __getitem__(self, index):
        img_file, label_file,  disp_file, calib_file, name = self.files[index]
        if self.cfg.IS_ISL_TRAINING and self.set == 'train':
            with open(label_file, 'rb') as f:
                semseg_label = np.load(f)
                if self.cfg.DEBUG:
                    semseg_label = cv2.resize(semseg_label, tuple(self.labels_size), interpolation=cv2.INTER_NEAREST)
        else:
            semseg_label = self.get_labels(label_file)
        image = self.get_image(img_file)
        depth_labels = self.get_depth_labels(disp_file, calib_file)
        image = self.preprocess(image)
        if self.cfg.IS_ISL_TRAINING and self.set == 'train':
            pass
        else:
            semseg_label = self.map_labels(semseg_label).copy()
        return image.copy(), semseg_label, depth_labels.copy(), np.array(image.shape), name