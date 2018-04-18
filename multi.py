import numpy as np
import torch.optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import multiprocessing
from models import linmodel
import utils
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


def torch_list_sum(list_):
    s = torch.stack(list_)
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
    return g, loss


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
    losses = []
    for i in range(num_iterations):
        optimizer.zero_grad()
        prediction = model(x)
        loss = linmodel.cost(y, prediction)
        loss.backward()
        optimizer.step()
        losses.append(loss.data.numpy())

    estimated = [param.data for param in model.parameters()]
    return losses, estimated, model

if __name__ == "__main__":

    NUM_ITERATIONS = 2500
    NUM_PARTITIONS = 10
    N = 500  # 50 - 500 - 1000 - 5000
    dx = 10  # log fino a 1M (0-6)
    dy = 5

    # torch variables
    x, y, w = init_data(N, dx, dy)
    model, optimizer = instantiate_model(dx, dy)
    t0 = time.time()
    mono_losses, mono_params, m = monolithic_run(x, y, model, optimizer, NUM_ITERATIONS)
    t1 = time.time()
    t_mono = (t1 - t0) * 1000
    print('monolithic run: {} msec'.format(t_mono))
    # print(w)
    # print(m.state_dict())

    mp.set_start_method('spawn')
    model, optimizer = instantiate_model(dx, dy)

    t0 = time.time()
    y_slices = list(torch.split(y, int(y.size()[1] / NUM_PARTITIONS), dim=1))
    x_slices = list(torch.split(x, int(x.size()[0] / NUM_PARTITIONS)))
    parts = [(i, j) for i, j in zip(x_slices, y_slices)]

    print('number of splits = {}'.format(len(parts)))
    model.share_memory()
    mngr = mp.Manager()

    num_processes = NUM_PARTITIONS

    new_gradient = None
    multi_losses = []
    with mp.Pool(processes=num_processes) as pool:
        for i in range(NUM_ITERATIONS):
            p = partial(run, model=model, optimizer=optimizer)
            res = pool.map(p, parts)
            new_grad = torch_list_sum([x[0] for x in res])
            loss = list(torch_list_sum([x[1] for x in res]).data.numpy())
            multi_losses.append(loss)
            model.linear.weight.grad = new_grad
            optimizer.step()

    t1 = time.time()
    t_multi = (t1 - t0) * 1000
    print('multiprocess run: {} msec'.format(t_multi))
    # print(model.state_dict())

    fig, ax = plt.subplots()
    ax.plot(mono_losses, label='monolithic')
    ax.plot(multi_losses, color='red', label='multiprocessing')
    plt.title('{}-{}-{}-{}'.format(N, dx, dy, NUM_PARTITIONS))
    plt.legend()
    plt.xlim(0, NUM_ITERATIONS)
    plt.show()
    plt.close()
