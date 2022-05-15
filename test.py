import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data as data
import torchvision.datasets as datasets

from net import *
from data_loader import *
from util import AverageMeter
import numpy as np
from PIL import Image
from collections import OrderedDict

parser = argparse.ArgumentParser(description='Test ArgumentParser')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='Number of Data Loading Workers (Default: 0)')
parser.add_argument('--batch_size', default=4, type=int, metavar='N',
                    help='Mini-batch Size (Default: 4)')
parser.add_argument('--print_freq', default=100, type=int, metavar='N',
                    help='Print Frequency (Default: 100)')
parser.add_argument('--load_ckpt', type=str, metavar='PATH',
                    help='Path to Load Checkpoint')
parser.add_argument('--save_dir', default='demo', type=str, metavar='PATH',
                    help='Path to Save Generated Frames (Default: demo)')
parser.add_argument('--gpu', default='0', type=str, help='cuda_visible_devices')
parser.add_argument('--amp_factor', default=20.0, type=float,
                    help='Amplification Factor (Default: 20.0)')
parser.add_argument('--mode', default='static', type=str, choices=['static', 'dynamic', 'temporal'],
                    help='Amplification Mode (choices: static, dynamic, temporal)')
parser.add_argument('--video_path', default='/demo_video/', type=str,
                    help='Path to Video Frames')
parser.add_argument('--num_data', default=300, type=int, help='Number of Frames')

# For Temporal Filter
parser.add_argument('--fh', default=0.4, type=float)
parser.add_argument('--fl', default=0.04, type=float)
# parser.add_argument('--fs', default=30, type=int)
# parser.add_argument('--ntab', default=2, type=int)

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)


def main():
    print(args)
    model = dmmm()
    model = model.cuda()

    # Load ckpt
    if os.path.isfile(args.load_ckpt):
        print("==> Loading Check Point '{}'".format(args.load_ckpt))
        ckpt = torch.load(args.load_ckpt)
        state_dict = ckpt['state_dict']
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            new_key = key[7:]
            new_state_dict[new_key] = value
        model.load_state_dict(new_state_dict)
        print("==> Load Check Point '{}' (epoch {})".format(args.load_ckpt, ckpt['epoch']))
    else:
        print("==> No Check Point File Found at '{}'".format(args.load_ckpt))
        assert False

    # ckpt Saving Directory
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(save_dir)

    # Data Loader
    dataset = DataFromFolderTest(root=args.video_path, mag=args.amp, mode=args.mode, num_data=args.num_data,
                                 preprocessing=False)
    data_loader = data.DataLoader(dataset,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=args.workers,
                                  pin_memory=True)

    model.eval()

    mag_frames = []

    if args.mode == 'static' or args.mode == 'dynamic':
        for i, (frame_a, frame_b, amp_factor) in enumerate(data_loader):
            if i % 10 == 0:
                print("Processing Sample %d" % i)
            amp_factor = amp_factor.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            frame_a = frame_a.cuda()
            frame_b = frame_b.cuda()
            amp_factor = amp_factor.cuda()

            y_hat, _, _, _ = model(frame_a, frame_b, frame_b, amp_factor)

            if i == 0:
                temp = frame_a.permute(0, 2, 3, 1).cpu().detach().numpy()
                temp = np.clip(temp, -1.0, 1.0)
                temp = ((temp + 1.0) * 127.5).astype(np.uint8)
                mag_frames.append(temp)

            y_hat = y_hat.permute(0, 2, 3, 1).cpu().detach().numpy()
            y_hat = np.clip(y_hat, -1.0, 1.0)
            y_hat = ((y_hat + 1.0) * 127.5).astype(np.uint8)
            mag_frames.append(y_hat)

    # else:
    #     filter_b = [args.fh - args.fl, args.fl - args.fh]
    #     filter_a = [-1.0 * (2.0 - args.fh - args.fl), (1.0 - args.fl) * (1.0 - args.fh)]

    mag_frames = np.concatenate(mag_frames, 0)
    for i, frame in enumerate(mag_frames):
        f_n = os.path.join(save_dir, 'demo_%s_%06d.png' % (args.mode, i))
        img = Image.fromarray(frame)
        img.save(f_n)



if __name__ == '__main__':
    main()
