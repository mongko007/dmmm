from torchvision.datasets.folder import *
import numpy as np
import torch


def add_poisson_noise(image):
    nn = np.random.uniform(0, 0.3)  # 0.3
    n = np.random.normal(0.0, 1.0, image.shape)
    n_str = np.sqrt(image + 1.0) / np.sqrt(127.5)
    return image + nn * n * n_str
