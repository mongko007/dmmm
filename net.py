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
        self.encoder = Encoder(3)
        self.decoder = Decoder()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def inference(self, xa, xb, alpha):
        xa_text_repr, xa_shape_repr = self.encoder(xa)
        xb_text_repr, xb_shape_repr = self.encoder(xb)
        manipulator_out = Manipulator(xa_shape_repr, xb_shape_repr, alpha)
        up_xb_text_repr = self.up(xb_text_repr)
        decoder_in = torch.cat([up_xb_text_repr, manipulator_out], dim=1)
        magnified_frame = self.decoder(decoder_in)

        return magnified_frame


model = dmmm()
print(model)
