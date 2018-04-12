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
        # w = np.random.randn(dx).reshape([dx, dy])
        w = np.random.randint(1, 10, size=(dx, dy))
        invertible = check_if_invertible(w)

    if w is not None:
        y = x.dot(w)
    else:
        exit('oops!')

    print(y.shape, x.shape, w.shape)

    # assert y.shape == (nsamples, dy)

    return x, y, w