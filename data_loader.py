from torchvision.datasets.folder import *
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from util import *


class DataFromFolder(ImageFolder):
    def __init__(self,
                 root,
                 img_num=3,
                 preprocessing=False,
                 transform=None,
                 target_transform=None,
                 loader=default_loader):
        amp_factors = np.loadtxt(os.path.join(root, 'train_mf.txt'))
        # print(amp_factors)
        # print(len(amp_factors))
        # print(amp_factors.dtype)
        # print(amp_factors.size)

        imgs = [(os.path.join(root, 'amplified', '%06d.png' % i),
                 os.path.join(root, 'frameA', '%06d.png' % i),
                 os.path.join(root, 'frameB', '%06d.png' % i),
                 os.path.join(root, 'frameC', '%06d.png' % i),
                 amp_factors[i])
                for i in range(img_num)]

        self.root = root
        self.preprocessing = preprocessing
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.imgs = imgs
        self.img_num = img_num

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

    def __len__(self):
        return self.img_num


class DataFromFolderTest(ImageFolder):
    def __init__(self,
                 root,
                 amp_factor=20.0,
                 mode='static',
                 img_num=300,
                 preprocessing=False,
                 transform=None,
                 target_transform=None,
                 loader=default_loader):

        # print(amp_factors)
        # print(len(amp_factors))
        # print(amp_factors.dtype)
        # print(amp_factors.size)

        if mode == 'static' or mode == 'temporal':
            imgs = [(root + '%06d.png' % 1,
                     root + '%06d.png' % (i + 2),
                     amp_factor) for i in range(img_num)]
        elif mode == 'dynamic':
            imgs = [(root + '%06d.png' % (i + 1),
                     root + '%06d.png' % (i + 2),
                     amp_factor) for i in range(img_num)]
        else:
            raise ValueError("Unsupported mode %s" % mode)

        self.root = root
        self.preprocessing = preprocessing
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.imgs = imgs
        self.img_num = img_num
        # self.samples = self.img

    def __getitem__(self, index):
        path_a, path_b, amp_factor = self.imgs[index]

        frame_a = np.array(self.loader(path_a))
        frame_b = np.array(self.loader(path_b))
        frame_a = frame_a / 127.5 - 1.0
        frame_b = frame_b / 127.5 - 1.0

        if self.preprocessing:
            frame_a = add_poisson_noise(frame_a)
            frame_b = add_poisson_noise(frame_b)

        frame_a = torch.from_numpy(frame_a)
        frame_b = torch.from_numpy(frame_b)
        frame_a = frame_a.float()
        frame_b = frame_b.float()
        frame_a = frame_a.permute(2, 0, 1)
        frame_b = frame_b.permute(2, 0, 1)

        amp_factor = torch.from_numpy(np.array(amp_factor)).float()

        return frame_a, frame_b, amp_factor

    def __len__(self):
        return self.img_num


if __name__ == '__main__':
    # dataset = DataFromFolder('data/train', img_num=3)
    # dataset = DataFromFolderTest('data/test/', mode='static', img_num=1)
    # dataset = DataFromFolderTest('data/test/', mode='temporal', img_num=1)
    dataset = DataFromFolderTest('data/test/', mode='dynamic', img_num=1)
    print(dataset)
    dataset.__getitem__(0)
    data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=2)
    print(data_loader)
