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


def run2(q, r, model, optimizer, num_iterations):
    name = multiprocessing.current_process().name
    # print('starting {}'.format(name))
    tmp = []
    if not q.empty():
        x, y = q.get()
        for i in range(num_iterations):
            optimizer.zero_grad()
            prediction = model(x)  # x = (400, 1): x1 = (200. 1). x2 = (200, 1)
            loss = linmodel.cost(y, prediction)
            loss.backward()  # get the gradients
            tmp.append(loss.data.numpy())
            optimizer.step()
        g = Variable([param.grad.data for param in model.parameters()][0])
        # print(name, len(tmp))
        to_return = {'g': g, 'loss': tmp, 'model': model.state_dict()}
        r.put(to_return)


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
    N = 5000  # 50 - 500 - 1000 - 5000
    dx = 100  # log fino a 1M (0-6)
    dy = 5

    # torch variables
    x, y, w = init_data(N, dx, dy)
    model, optimizer = instantiate_model(dx, dy)
    t0 = time.time()
    mono_losses, mono_param_l, m = monolithic_run(x, y, model, optimizer, NUM_ITERATIONS)

    mono_params = Variable(mono_param_l[0])
    print(type(mono_params), mono_params.size())
    # print(mono_losses[-1])
    t1 = time.time()
    t_mono = (t1 - t0) * 1000
    print('monolithic run: {} msec'.format('%.2f' % t_mono))
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

    q = mngr.Queue(maxsize=NUM_PARTITIONS)
    r = mngr.Queue(maxsize=NUM_PARTITIONS)
    for el in parts:
        q.put(el)
        time.sleep(0.1)
    print('input queue full: {}'.format(q.full()))

    num_processes = NUM_PARTITIONS

    processes = []
    multi_losses = []
    for rank in range(num_processes):
        p = mp.Process(target=run2, args=(q, r, model, optimizer, NUM_ITERATIONS))
        p.start()
        processes.append(p)

    print('started {} processes'.format(len(processes)))

    for p in processes:
        p.join()

    results = []
    while not r.empty():
        results.append(r.get())

    gradient = torch_list_sum([x['g'] for x in results])
    multi_losses = np.sum([x['loss'] for x in results], axis=0)

    models_dicts = [Variable(x['model']['linear.weight']) for x in results]
    for pos, estimated_param in enumerate(models_dicts):
        # print(pos, torch.sum((estimated_param.mm(x.t()) - y)**2).data.numpy())
        print(pos, torch.sum((estimated_param - mono_params)**2).data.numpy())

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
