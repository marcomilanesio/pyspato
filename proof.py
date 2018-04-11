import numpy as np
import sys
import torch
from torch.autograd import Variable


THRESHOLD = 1e-5  # max error


def check_if_invertible(m):
    return np.linalg.cond(m) < 1 / sys.float_info.epsilon


def init_data(nsamples, dx, dy=1):
    x = np.random.randn(nsamples * dx).reshape((nsamples, dx))
    w = None
    y = None

    invertible = False
    while not invertible:
        w = np.random.randn(dx).reshape([dx, dy])
        invertible = check_if_invertible(w)

    if w is not None:
        y = x.dot(w)
    else:
        exit('oops!')

    assert y.shape == (nsamples, dy)

    # print(y.shape, x.shape, w.shape)
    return x, y, w


def closed_form(x, y, w, splits=2):
    print(x.shape, y.shape, w.shape)
    E = np.sum((y - x.dot(w))**2)

    mini_sums = []
    y_slices = np.array_split(y, splits)
    x_slices = np.array_split(x, splits)
    assert len(x_slices) == len(y_slices)

    for i, yslice in enumerate(y_slices):
        tmp_sum = np.sum((yslice - x_slices[i].dot(w))**2)
        mini_sums.append(tmp_sum)

    resid = np.sum(mini_sums) - E
    t1 = resid < THRESHOLD

    grad = -2 * x.T.dot(y - x.dot(w))
    mini_grads = []
    for i, yslice in enumerate(y_slices):
        tmp_grad = -2 * x_slices[i].T.dot(yslice - x_slices[i].dot(w))
        mini_grads.append(tmp_grad)

    resid = np.sum(mini_grads) - grad
    t2 = np.all([el < THRESHOLD for el in resid])
    return t1 and t2


def convert_to_variable(mx, my, mw):
    x = Variable(torch.from_numpy(mx), requires_grad=False).type(torch.FloatTensor)
    y = Variable(torch.from_numpy(my), requires_grad=False).type(torch.FloatTensor)
    w = Variable(torch.from_numpy(mw), requires_grad=False).type(torch.FloatTensor)
    return x, y, w


def closed_form_torch_to_numpy(x, y, w):
    return closed_form(x.data.numpy(), y.data.numpy(), w.data.numpy())


def closed_form_torch(x, y, w, splits=2):
    # print('x = ', x.size(), '; y = ', y.size(), '; w =', w.size())
    grad = -2 * x.t().mm(y - x.mm(w))

    y_slices = list(torch.split(y, int(y.size()[0] / splits)))
    x_slices = list(torch.split(x, int(x.size()[0] / splits)))

    # print('y_slices: ', [a.size() for a in y_slices])
    # print('x_slices: ', [a.size() for a in x_slices])
    grads = []
    for i, yslice in enumerate(y_slices):
        tmp_grad = -2 * x_slices[i].t().mm(yslice - x_slices[i].mm(w))
        grads.append(tmp_grad)

    s = torch.stack(grads)
    su = torch.sum(s, dim=0)

    resid = su - grad
    return np.all([el < THRESHOLD for el in resid.data.numpy()])

if __name__ == '__main__':
    n = 100000
    dx = 100000
    x, y, w = init_data(n, dx)
    print(closed_form(x, y, w, splits=2))
    x1, y1, w1 = convert_to_variable(x, y, w)
    print(closed_form_torch(x1, y1, w1))
