from torchvision.datasets.folder import *
import numpy as np
import torch


def add_poisson_noise(image):
    nn = np.random.uniform(0, 0.3)  # 0.3
    n = np.random.normal(0.0, 1.0, image.shape)
    n_str = np.sqrt(image + 1.0) / np.sqrt(127.5)
    return image + nn * n * n_str


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0
        self.val = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count
