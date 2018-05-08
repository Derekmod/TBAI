from __future__ import print_function

import torch
import torch.nn.functional as F

from qmath import *


def gameActivation(input):
    # break into v and u
    # apply sigmoid to v
    # apply softmax to u

    splits = input.split(1,1)
    #print(splits)
    #left, right = input.split(2, 1)

    left, right = splits[0], splits[1]

    value = torch.sigmoid(left)
    uncertainty = F.softplus(right)

    return torch.cat((value, uncertainty), 1)

class TBAINN(torch.nn.module):
    def __init__(self, qlog_growth_pow=0.1, qlog_reference_epoch=1000., start_epoch=0, start_q_choice=1.):
        self.qlog_growth_pow = qlog_growth_pow
        self.qlog_reference_epoch = qlog_reference_epoch
        self.epoch = start_epoch
        self.start_q_choice = start_q_choice

    @property
    def q_choice(self):
        beta = qlog(self.qlog_reference_epoch + 1, self.qlog_growth_pow)
        base = 1. - qlog(self.epoch + 1, self.qlog_growth_pow) / beta
        return self.start_q_choice * base