import numpy as np


def standardize(x):
    return (x - np.mean(x)) / np.std(x)


def add_noise(y):
    rnd = np.random.randn(y.shape[0])
    return y + rnd