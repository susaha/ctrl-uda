import cv2
import numpy as np
import torchvision.transforms as transforms


def get_depth_dada(dataset, file, phase=None):
    depth = cv2.imread(str(file), flags=cv2.IMREAD_ANYDEPTH).astype(np.float32)
    depth = cv2.resize(depth, tuple(dataset.labels_size), interpolation=cv2.INTER_NEAREST)
    depth = 65536.0 / (depth + 1)  # inverse depth
    return depth


def get_depth_gasda(dataset, file, phase=None):
    if not phase:
        raise NotImplementedError('phase value is none!!')
    depth = cv2.imread(str(file), flags=cv2.IMREAD_ANYDEPTH).astype(np.float32)
    depth = cv2.resize(depth, tuple(dataset.labels_size), interpolation=cv2.INTER_NEAREST)
    if phase == 'test':
        toTensor = transforms.ToTensor()
        depth = toTensor(depth)
        return depth
    elif phase == 'train':
        depth = np.array(depth, dtype=np.float32)
        depth /= 65536.0
        depth[depth < 0.0] = 0.0
        depth = depth * 2.0
        depth -= 1.0
        return depth