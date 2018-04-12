import numpy as np
import sys
import torch
from torch.autograd import Variable

THRESHOLD = 1e-10  # max error


def check_if_invertible(m):
    return np.linalg.cond(m) < 1 / sys.float_info.epsilon


def init_data(nsamples, dx, dy=1):
    x = np.random.randn(nsamples * dx).reshape((nsamples, dx))
    w = None
    y = None

    invertible = False
    while not invertible:
        w = np.random.randn(dx).reshape([dy, dx])
        invertible = check_if_invertible(w)

    if w is not None:
        y = w.dot(x.T)
    else:
        exit('oops!')

    print(y.shape, x.shape, w.shape)
    return x, y, w


def cost_fn(x, y, w):
    return np.sum((y - w.dot(x.T))**2)


def grad_closed_form(x, y, w):
    """
    :param x: N, dx
    :param y: 1, N
    :param w: 1, dx
    :return:
    """
    # print(x.shape, y.shape, w.shape)
    return -2 * x.T.dot(y.T - x.dot(w.T))


def closed_form(x, y, w, splits=2):
    print('y:', y.shape, 'x:', x.shape, 'w:', w.shape)
    E = cost_fn(x, y, w)
    mini_sums = []
    y_slices = np.array_split(y, splits, axis=1)
    print('y split in ', len(y_slices), [a.shape for a in y_slices])
    x_slices = np.array_split(x, splits)
    print('x split in ', len(x_slices), [a.shape for a in x_slices])
    assert len(x_slices) == len(y_slices)

    for i, yslice in enumerate(y_slices):
        xslice = x_slices[i]
        tmp_sum = cost_fn(xslice, yslice, w)
        # tmp_sum = np.sum((yslice - x_slices[i].dot(w))**2)
        mini_sums.append(tmp_sum)

    resid = np.sum(mini_sums) - E
    t1 = resid < THRESHOLD

    grad = grad_closed_form(x, y, w)
    print('np grad', grad.shape)
    mini_grads = []
    for i, yslice in enumerate(y_slices):
        xslice = x_slices[i]
        g = grad_closed_form(xslice, yslice, w)
        mini_grads.append(g)

    print('np grad split in ', len(mini_grads), [g.shape for g in mini_grads])
    res = np.sum(mini_grads)
    # res = np.concatenate(mini_grads, axis=1)
    # print(res.shape)
    # resid = np.sum(mini_grads) - grad
    resid = res - grad
    # print(resid)
    t2 = np.all([el < THRESHOLD for el in resid])
    return t1 and t2


def convert_to_variable(mx, my, mw):
    x = Variable(torch.from_numpy(mx), requires_grad=False).type(torch.FloatTensor)
    y = Variable(torch.from_numpy(my), requires_grad=False).type(torch.FloatTensor)
    w = Variable(torch.from_numpy(mw), requires_grad=False).type(torch.FloatTensor)
    return x, y, w


def closed_form_torch_to_numpy(x, y, w):
    return closed_form(x.data.numpy(), y.data.numpy(), w.data.numpy())


def torch_grad(x, y, w):
    """

    :param x:N,dx
    :param y:1,N
    :param w:1,dx
    :return:
    """
    # print(x.size(), y.size(), w.size())
    return -2 * x.t().mm(y.t() - x.mm(w.t()))


def closed_form_torch(x, y, w, splits=2):
    # print('x = ', x.size(), '; y = ', y.size(), '; w =', w.size())
    grad = torch_grad(x, y, w)
    print('grad', grad.size())
    y_slices = list(torch.split(y, int(y.size()[1] / splits), dim=1))
    x_slices = list(torch.split(x, int(x.size()[0] / splits)))

    print('y_slices: ', [a.size() for a in y_slices])
    print('x_slices: ', [a.size() for a in x_slices])
    grads = []
    for i, yslice in enumerate(y_slices):
        xslice = x_slices[i]
        # tmp_grad = -2 * x_slices[i].t().mm(yslice - x_slices[i].mm(w))
        tmp_grad = torch_grad(xslice, yslice, w)
        print('tmp_grad', tmp_grad.size())
        grads.append(tmp_grad)

    s = torch.stack(grads)
    su = torch.sum(s, dim=0)

    resid = su - grad
    return np.all([el < THRESHOLD for el in resid.data.numpy()])

if __name__ == '__main__':
    n = 500
    dx = 10
    x, y, w = init_data(n, dx)
    print('closed form Numpy', closed_form(x, y, w, splits=2))
    x1, y1, w1 = convert_to_variable(x, y, w)
    print(closed_form_torch(x1, y1, w1))
