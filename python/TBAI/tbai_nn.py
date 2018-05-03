from __future__ import print_function

import torch
import torch.nn.functional as F


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