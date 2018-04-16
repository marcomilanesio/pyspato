import torch
import torch.nn as nn
from proof import torch_grad


def cost(target, predicted):
    """
    :param target: Y
    :param predicted: model(x)
    :return:
    """
    assert target.t().size() == predicted.size()
    # cost = torch.sum((torch.t(target) - predicted) ** 2)
    cost = torch.sum((target.t() - predicted) ** 2)
    # print(cost)
    return cost


# def mse(input, target):
#     return nn.functional.mse_loss(input, target)


class LinModel(nn.Module):
    def __init__(self, in_size, out_size):
        super(LinModel, self).__init__()  # always call parent's init
        self.linear = nn.Linear(in_size, out_size, bias=False)  # layer parameters

    def forward(self, x):
        out = self.linear(x)
        return out

    def backward(self, x, y, w):
        return -2 * x.t().mm(y.t() - x.mm(w.t()))