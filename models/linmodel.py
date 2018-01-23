import torch
import torch.nn as nn


def cost(target, predicted):
    """
    :param target: Y
    :param predicted: model(x)
    :return:
    """
    cost = torch.sum((torch.t(target) - predicted) ** 2)
    # print(cost)
    return cost


class LinModel(nn.Module):
    def __init__(self, in_size, out_size):
        super(LinModel, self).__init__()  # always call parent's init
        self.linear = nn.Linear(in_size, out_size, bias=False)  # layer parameters

    def forward(self, x):
        return self.linear(x)
