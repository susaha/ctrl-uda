import numpy as np
from ctrl.dataset.base_dataset import BaseDataset
from ctrl.dataset.depth import get_depth_dada, get_depth_gasda
import os


class SYNTHIADataSetDepth(BaseDataset):
    def __init__(
        self,
        root,
        list_path,
        set="all",
        num_classes=16,
        max_iters=None,
        crop_size=(321, 321),
        mean=(128, 128, 128),
        use_depth=False,
        depth_processing='GASDA',
        cfg=None,
        joint_transform=None,
    ):
        super().__init__(root, list_path, set, max_iters, crop_size, None, mean, joint_transform, cfg)

        if num_classes == 16:
            self.id_to_trainid = {
                3: 0,
                4: 1,
                2: 2,
                21: 3,
                5: 4,
                7: 5,
                15: 6,
                9: 7,
                6: 8,
                1: 9,
                10: 10,
                17: 11,
                8: 12,
                19: 13,
                12: 14,
                11: 15,
            }
        elif num_classes == 7:
            self.id_to_trainid = {
                1:4,
                2:1,
                3:0,
                4:0,
                5:1,
                6:3,
                7:2,
                8:6,
                9:2,
                10:5,
                11:6,
                15:2,
                22:0}
        else:
            raise NotImplementedError(f"Not yet supported {num_classes} classes")

        self.cfg = cfg
        self.joint_transform = joint_transform
        self.use_depth = use_depth
        self.depth_processing = depth_processing
        if self.use_depth:
            for (i, file) in enumerate(self.files):
                img_file, label_file, name = file
                depth_file = self.root / "Depth" / name
                self.files[i] = (img_file, label_file, depth_file, name)
            os.environ["MKL_NUM_THREADS"] = "1"
            os.environ["OMP_NUM_THREADS"] = "1"
            print('ctrl/dataset/synthia.py -->  __init__()')

    def get_metadata(self, name, mode=None):
        label_file = self.root / "parsed_LABELS" / name
        if mode == 'original_only':
            img_file = self.root / "RGB" / name
            return img_file, label_file
        elif mode == 'original_and_translated':
            img_file1 = self.root / "RGB" / name
            img_file2 = self.root / "SynthiaToCityscapesRGBs/Rui/images" / name
            return img_file1, img_file2, label_file
        elif mode == 'translated_only':
            img_file = self.root / "SynthiaToCityscapesRGBs/Rui/images" / name
            return img_file, label_file
        else:
            print('ctrl/dataset/synthia.py --> set proper value for cfg.SYNTHIA_DATALOADING_MODE')
            raise NotImplementedError

    def __getitem__(self, index):
        depth_file = None
        if self.use_depth:
            img_file, label_file, depth_file, name = self.files[index]
        else:
            img_file, label_file, name = self.files[index]
        image = self.get_image(img_file)
        label = self.get_labels(label_file)
        depth = None
        if self.use_depth:
            depth = self.get_depth(depth_file)
        # re-assign labels to match the format of Cityscapes
        label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        image = self.preprocess(image)
        image = image.copy()
        label_copy = label_copy.copy()
        shape = np.array(image.shape)
        depth = depth.copy()
        if self.use_depth:
            return image, label_copy, depth, shape, name
        else:
            return image, label_copy, shape, name

    def get_depth(self, file):
        if self.depth_processing == 'GASDA':
            return get_depth_gasda(self, file, phase='train')
        elif self.depth_processing == 'DADA':
            return get_depth_dada(self, file, phase='train')
