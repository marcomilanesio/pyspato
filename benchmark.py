import numpy as np
import main
import time
import matplotlib.pyplot as plt

# fieldnames = ['n_iterations', 'dx', 'mse', 'time']

n_iterations = 5000
n_samples = [50, 500, 1000, 5000]
dx = np.power(10, range(1, 7))
dy = 5

results = []


def plot_loss(losses, lossfname):
    fig, ax = plt.subplots()
    ax.plot(losses)
    # plt.ylim(0, 10000)
    # plt.xlim(0, 10000)
    fig.savefig(lossfname)
    plt.close()


def plot_scatter(model, in_, out_, scatterfname):
    pred = model(in_)
    fig, ax = plt.subplots()
    ax.scatter(pred.data.numpy(), out_.data.numpy())
    fig.savefig(scatterfname)
    plt.close()


for N in [50, 500, 1000, 5000]:
    for dx in [1, 10, 100, 1000, 10000, 100000]:
        lossfname = "./plots/loss-{}-{}.png".format(N, dx)
        scatterfname = "./plots/scatter-{}-{}.png".format(N, dx)
        print("running ", N, dx)
        start = time.time()
        x, y, w = main.init_data(N, dx, dy)
        m, o = main.instantiate_model(dx, dy)
        losses = []
        for i in range(n_iterations):
            m, o, l = main.step(x, y, m, o)
            losses.append(l)

        plot_loss(losses, lossfname)
        plot_scatter(m, x, y, scatterfname)

