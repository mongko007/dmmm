from torchvision.datasets.folder import *
import numpy as np
import torch
from util import *


class DataFromFolder(ImageFolder):
    def __init__(self,
                 path,
                 img_num=3,
                 preprocessing=None,
                 transform=None,
                 target_transform=None,
                 loader=default_loader):
        amp_factors = np.loadtxt(os.path.join(path, 'train_mf.txt'))
        # print(amp_factors)
        # print(len(amp_factors))
        # print(amp_factors.dtype)
        # print(amp_factors.size)

        imgs = [(os.path.join(path, 'amplified', '%06d.png' % i),
                 os.path.join(path, 'frameA', '%06d.png' % i),
                 os.path.join(path, 'frameB', '%06d.png' % i),
                 os.path.join(path, 'frameC', '%06d.png' % i),
                 amp_factors[i])
                for i in range(img_num)]

        self.path = path
        self.preprocessing = preprocessing
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.imgs = imgs

    def __getitem__(self, item_index):
        path_amplified, path_frameA, path_frameB, path_frameC, amp_factor = self.imgs[item_index]
        img_amplified = np.array(self.loader(path_amplified))
        img_frameA = np.array(self.loader(path_frameA))
        img_frameB = np.array(self.loader(path_frameB))
        img_frameC = np.array(self.loader(path_frameC))

        # Normalize
        img_amplified = img_amplified / 127.5 - 1.0
        img_frameA = img_frameA / 127.5 - 1.0
        img_frameB = img_frameB / 127.5 - 1.0
        img_frameC = img_frameC / 127.5 - 1.0

        # Add Poisson Noise
        if self.preprocessing:
            img_amplified = add_poisson_noise(img_amplified)
            img_frameA = add_poisson_noise(img_frameA)
            img_frameB = add_poisson_noise(img_frameB)
            img_frameC = add_poisson_noise(img_frameC)

        # Numpy to Tensor
        img_amplified = torch.from_numpy(img_amplified).float()
        img_frameA = torch.from_numpy(img_frameA).float()
        img_frameB = torch.from_numpy(img_frameB).float()
        img_frameC = torch.from_numpy(img_frameC).float()
        amp_factor = torch.from_numpy(np.array(amp_factor)).float()
        # print(img_frameA)
        # print(img_frameA.shape)

        # Permute from H-W-C to C-H-W
        img_amplified = img_amplified.permute(2, 0, 1)
        img_frameA = img_frameA.permute(2, 0, 1)
        img_frameB = img_frameB.permute(2, 0, 1)
        img_frameC = img_frameC.permute(2, 0, 1)
        # print(img_frameA.shape)

        return img_amplified, img_frameA, img_frameB, img_frameC, amp_factor


if __name__ == '__main__':
    dataset = DataFromFolder('data/train')
    dataset.__getitem__(0)
