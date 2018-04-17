import numpy as np
import torch.optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import multiprocessing
from models import linmodel
import utils
import numpy.linalg as linalg
import sys
import concurrent.futures
import threading
import time
import torch.multiprocessing as mp
from functools import partial


def init_data(nsamples, dx, dy):
    X, Y, W = utils.init_data(nsamples, dx, dy)

    x = Variable(torch.from_numpy(X), requires_grad=True).type(torch.FloatTensor)
    y = Variable(torch.from_numpy(Y), requires_grad=True).type(torch.FloatTensor)
    w = Variable(torch.from_numpy(W), requires_grad=True).type(torch.FloatTensor)
    return x, y, w


def instantiate_model(dx, dy):
    model = linmodel.LinModel(dx, dy)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    return model, optimizer


def gradients_sum(gradients):
    # print('gradients: {}'.format(gradients))
    s = torch.stack(gradients)
    su = torch.sum(s, dim=0)
    return su


def run(parts, model, optimizer):
    name = multiprocessing.current_process().name
    x, y = parts
    optimizer.zero_grad()
    prediction = model(x)  # x = (400, 1): x1 = (200. 1). x2 = (200, 1)
    loss = linmodel.cost(y, prediction)
    loss.backward()  # get the gradients
    g = Variable([param.grad.data for param in model.parameters()][0])
    # print(name, g)
    return g


def plot_loss(loss, fname=False):
    fig, ax = plt.subplots()
    ax.plot(loss)
    # plt.ylim(0, 10000)
    plt.xlim(0, NUM_ITERATIONS)
    if fname:
        fig.savefig(fname)
    else:
        plt.show()
    plt.close()


def monolithic_run(x, y, model, optimizer, num_iterations):
    # criterion = torch.nn.MSELoss()
    t0 = time.time()
    losses = []
    for i in range(num_iterations):
        # m, o, l = step(x, y, m, o)
        optimizer.zero_grad()
        prediction = model(x)
        loss = linmodel.cost(y, prediction)
        # loss = criterion(prediction, y)

        loss.backward()
        optimizer.step()
        # print('mono param', [param.data for param in m.parameters()])
        # print('mono grad', [param.grad.data for param in m.parameters()])
        losses.append(loss.data.numpy())

    t1 = time.time()
    # print('monolithic run done in {} msec'.format("%.2f" % (1000 * (t1 - t0))))
    estimated = [param.data for param in model.parameters()]
    return losses, estimated, model

if __name__ == "__main__":

    NUM_ITERATIONS = 2500
    NUM_PARTITIONS = 2
    N = 500  # 50 - 500 - 1000 - 5000
    dx = 5  # log fino a 1M (0-6)
    dy = 5

    # torch variables
    x, y, w = init_data(N, dx, dy)
    model, optimizer = instantiate_model(dx, dy)
    losses, mono_params, m = monolithic_run(x, y, model, optimizer, NUM_ITERATIONS)
    plot_loss(losses, False)

    print(w)
    print(m.state_dict())

    mp.set_start_method('spawn')
    model, optimizer = instantiate_model(dx, dy)

    y_slices = list(torch.split(y, int(y.size()[1] / NUM_PARTITIONS), dim=1))
    x_slices = list(torch.split(x, int(x.size()[0] / NUM_PARTITIONS)))
    parts = [(i, j) for i, j in zip(x_slices, y_slices)]

    print('number of splits = {}'.format(len(parts)))
    model.share_memory()
    mngr = mp.Manager()

    num_processes = NUM_PARTITIONS

    new_gradient = None

    with mp.Pool(processes=num_processes) as pool:
        for i in range(NUM_ITERATIONS):
            p = partial(run, model=model, optimizer=optimizer)
            res = pool.map(p, parts)
            # print('got', len(res), type(res[0]))
            g = gradients_sum(res)
            model.linear.weight.grad = g
            optimizer.step()

    print(model.state_dict())

    # fig, axes = plt.subplots(5, 2, sharex=True, sharey=True)
    # i, j = 0, 0
    # for name, lst in losses.items():
    #     axes[i % 5, j % 2].plot(lst, label='{}'.format(name))
    #     legend = axes[i % 5, j % 2].legend(loc='upper right')
    #     plt.legend()
    #     plt.ylim(0, 3000)
    #     plt.xlim(0, 500)
    #     i += 1
    #     j += 1
    # fig.savefig('./plots/process_loss.png')
    # for k, v in losses.items():
    #     print(k, v[-2:])


