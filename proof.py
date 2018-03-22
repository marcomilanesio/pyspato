import numpy as np
import sys


THRESHOLD = 1e-12  # max error

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

    print(y.shape, x.shape, w.shape)
    return x, y, w


def closed_form(x, y, w, splits=2):
    # sum
    E = np.sum((y - x.dot(w))**2)

    mini_sums = []
    y_slices = np.array_split(y, splits)
    x_slices = np.array_split(x, splits)
    assert len(x_slices) == len(y_slices)

    for i, yslice in enumerate(y_slices):
        tmp_sum = np.sum((yslice - x_slices[i].dot(w))**2)
        mini_sums.append(tmp_sum)

    resid = np.sum(mini_sums) - E
    print('Sum: ', resid < THRESHOLD)

    # gradients
    grad = -2 * x.T.dot(y - x.dot(w))
    mini_grads = []
    for i, yslice in enumerate(y_slices):
        tmp_grad = -2 * x_slices[i].T.dot(yslice - x_slices[i].dot(w))
        mini_grads.append(tmp_grad)

    resid = np.sum(mini_grads) - grad
    print('Grad:', np.all([el < THRESHOLD for el in resid]))


if __name__ == '__main__':
    n = 100
    dx = 10
    x, y, w = init_data(n, dx)
    closed_form(x, y, w, splits=2)

