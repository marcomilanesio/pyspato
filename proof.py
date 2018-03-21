import numpy as np
import sys


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

    if w:
        y = x.dot(w)
    else:
        exit('oops!')

    assert y.shape == (nsamples, dy)

    print(y.shape, x.shape, w.shape)
    return x, y, w


def closed_form(x, y, w, splits=2):
    E = np.sum((y - x.dot(w))**2)
    print(E)
    grad = -2 * x.T.dot(y - x.dot(w))
    print(grad)

    slice_dim = y.shape[0] / splits
    y1 = y[50:,]
    y2 = y[:50,]
    x1 = x[50:,:]
    x2 = x[:50,:]

    #print(y1.shape, x1.shape, w.shape)
    grad1 = -2 * x1.T.dot(y1 - x1.dot(w))
    grad2 = -2 * x2.T.dot(y2 - x2.dot(w))
    print(grad1)
    print(grad2)
    print(all((grad1 + grad2) - grad) < 1e-12)


if __name__ == '__main__':
    n = 100
    dx = 10
    x, y, w = init_data(n, dx)