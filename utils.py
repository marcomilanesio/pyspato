import numpy as np
import sys


def standardize(x):
    """

    :param x: np.array
    :return: standardized vector (mu=0, std=1)
    """
    return (x - np.mean(x)) / np.std(x)


def add_noise(y):
    rnd = np.random.randn(y.shape[0])
    return y + rnd


def check_if_invertible(m):
    return np.linalg.cond(m) < 1 / sys.float_info.epsilon


def init_data(nsamples, dx, dy=1):
    # x = np.random.randn(nsamples * dx).reshape((nsamples, dx))
    x = standardize(np.linspace(0, 1000, nsamples * dx).reshape((nsamples, dx)))
    w = None
    y = None

    invertible = False
    while not invertible:
        # w = np.random.randn(dy * dx).reshape([dy, dx])
        w = np.random.randint(1, 10, size=(dy, dx))
        invertible = check_if_invertible(w)

    if w is not None:
        y = w.dot(x.T)
    else:
        exit('oops!')

    print('Created y: {}, x: {}, w: {}'.format(y.shape, x.shape, w.shape))
    return x, y, w