import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

from module import *


class dmmm(nn.Module):
    def __init__(self):
        super(dmmm, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.manipulator = Manipulator()

    def inference(self, x_a, x_b, x_c, amp_factor):
        xa_text_repr, xa_shape_repr = self.encoder(x_a)
        xb_text_repr, xb_shape_repr = self.encoder(x_b)
        xc_text_repr, xc_shape_repr = self.encoder(x_c)

        manipulator_out = self.manipulator(xa_shape_repr, xb_shape_repr, amp_factor)
        magnified_frame = self.decoder(xb_text_repr, manipulator_out)

        return magnified_frame, (xa_text_repr, xa_shape_repr), (xb_text_repr, xb_shape_repr), (xc_text_repr, xc_shape_repr)


model = dmmm()
print(model)
