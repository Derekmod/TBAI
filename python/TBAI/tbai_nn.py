import torch


def gameActivation(input):
    # break into v and u
    # apply sigmoid to v
    # apply softmax to u
    left, right = input.split(2, 1)

    value = torch.sigmoid(left)
    uncertainty = torch.softplus(right)

    return torch.cat((value, uncertainty), 1)