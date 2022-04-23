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
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from net import dmmm
from data_loader import DataFromFolder
from util import *

parser = argparse.ArgumentParser(description='Arguments of dmmm')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=12, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch_size', default=4, type=int, metavar='N',
                    help='mini-batch size (default: 4)')
parser.add_argument('--lr', default=0.0001, type=float, metavar='LR',
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', default=0.0, type=float, metavar='W',
                    help='weight decay (default: 0.0)')
parser.add_argument('--num_data', default=100000, type=int,
                    help='number of total data sample used for training (default: 100000)')
# parser.add_argument('--resume', default='', type=str, metavar='PATH',
#                     help='path to latest checkpoint (default: none)')
parser.add_argument('--ckpt', default='ckpt', type=str, metavar='PATH',
                    help='path to save checkpoint (default: ckpt)')
parser.add_argument('--weight_loss_1', default=1.0, type=float,
                    help='weight texture regularization loss  (default: 1.0)')
parser.add_argument('--weight_loss_3', default=1.0, type=float,
                    help='weight shape regularization loss  (default: 1.0)')
parser.add_argument('--gpu', default=0, type=str, help='cuda_visible_devices')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

losses_0, losses_1, losses_2, losses_3 = [], [], [], []


def main():
    global args
    args = parser.parse_args()
    print(args)

    devices_id = [0, 1]
    model = dmmm()
    model = torch.nn.DataParallel(model, devices_id).cuda()
    cudnn.benchmark = True

    # Dataloader
    dataset = DataFromFolder('data/train', img_num=args.num_data, preprocessing=True)
    data_loader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    # Loss Criterion
    loss_criterion = nn.L1Loss(size_average=True).cuda()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 args.lr,
                                 betas=(0.9, 0.999),
                                 weight_decay=args.weight_decay)

    # Check Saving Directory
    ckpt_dir = args.ckpt
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    print(ckpt_dir)

    # Training Model
    for epoch in range(0, args.epochs):
        current_loss_0_avg, current_loss_1_avg, current_loss_2_avg, current_loss_3_avg = \
            train(model, data_loader, epoch, loss_criterion, optimizer, args)

        losses_0.append(current_loss_0_avg)
        losses_1.append(current_loss_1_avg)
        losses_2.append(current_loss_2_avg)
        losses_3.append(current_loss_3_avg)

        checkpoint_dict = {
            'Epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'losses_0': losses_0,
            'losses_1': losses_1,
            'losses_2': losses_2,
            'losses_3': losses_3
        }

        # Save Checkpoints
        ckpt_path = os.path.join(ckpt_dir, 'ckpt_e%02d.pth' % epoch)
        torch.save(checkpoint_dict, ckpt_path)


def l1_loss(input, target):
    return torch.abs(input - target).mean()


def train(model, data_loader, epoch, criterion, optimizer, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    current_loss_0 = AverageMeter()
    current_loss_1 = AverageMeter()  # Texture Loss
    current_loss_2 = AverageMeter()
    current_loss_3 = AverageMeter()  # Shape Loss

    model.train()

    for i, (img_amplified, img_frameA, img_frameB, img_frameC, amp_factor) in enumerate(data_loader):
        amp_factor = amp_factor.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        magnified_frame, (xa_text_repr, xa_shape_repr), (xb_text_repr, xb_shape_repr), (xc_text_repr, xc_shape_repr) = \
            model(img_frameA, img_frameB, img_frameC, amp_factor)
        y = img_amplified
        y_hat = magnified_frame
        v_a, m_a = (xa_text_repr, xa_shape_repr)
        v_b, m_b = (xb_text_repr, xb_shape_repr)
        v_c, m_c = (xc_text_repr, xc_shape_repr)

        loss_y_y_hat = criterion(y, y_hat)
        loss_1 = args.weight_loss_1 * l1_loss(v_a, v_c)
        loss_2 = 0.0
        loss_3 = args.weight_loss_3 * l1_loss(m_c, m_b)
        loss = loss_y_y_hat + loss_1 + loss_2 + loss_3

        current_loss_0.update(loss_y_y_hat.item())
        current_loss_1.update(loss_1.item())
        current_loss_2.update(loss_2.item())
        current_loss_3.update(loss_3.item())

        # Update Model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Loss_1 {loss_1.val:.4f} ({loss_1.avg:.4f})\t'
                  'Loss_2 {loss_2.val:.4f} ({loss_2.avg:.4f})\t'
                  'Loss_3 {loss_3.val:.4f} ({loss_3.avg:.4f})'.format(
                epoch, i, len(data_loader), batch_time=batch_time, data_time=data_time,
                loss=current_loss_0, loss_1=current_loss_1,
                loss_2=current_loss_2, loss_3=current_loss_3))

        return current_loss_0.avg, current_loss_1.avg, current_loss_2.avg, current_loss_3.avg
