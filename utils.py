import numpy as np


def standardize(x):
    """

    :param x: np.array
    :return: standardized vector (mu=0, std=1)
    """
    return (x - np.mean(x)) / np.std(x)


def add_noise(y):
    rnd = np.random.randn(1)
    return y + rnd