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


def init_data(nsamples, dx, dy):
    Xold = np.linspace(0, 1000, nsamples * dx).reshape([nsamples, dx])
    X = utils.standardize(Xold)

    invertible = False
    while not invertible:
        W = np.random.randint(1, 10, size=(dy, dx))
        if linalg.cond(W) < 1 / sys.float_info.epsilon:
            invertible = True
            print('W invertible')

    Y = W.dot(X.T)  # target

    # for i in range(Y.shape[1]):
    #     Y[:, i] = utils.add_noise(Y[:, i])

    x = Variable(torch.from_numpy(X), requires_grad=True).type(torch.FloatTensor)
    y = Variable(torch.from_numpy(Y), requires_grad=True).type(torch.FloatTensor)
    w = Variable(torch.from_numpy(W), requires_grad=True).type(torch.FloatTensor)
    return x, y, w


def instantiate_model(dx, dy):
    model = linmodel.LinModel(dx, dy)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    return model, optimizer


def step(x, y, model, optimizer, g=None):
    prediction = model(x)  # x = (400, 1): x1 = (200. 1). x2 = (200, 1)
    loss = linmodel.cost(y, prediction)
    optimizer.zero_grad()
    loss.backward()  # get the gradients
    # if g is not None:
    #     model.linear.weight.grad = g
    # print([param.grad.data for param in model.parameters()])
    # sum gradients
    optimizer.step()  #
    # print('loss', loss)
    return model, optimizer, loss.data.numpy()


# def run(x, y, m, o):
#     name = threading.current_thread().name
#     print('{} working on {}-{}'.format(name, x.size(), y.size()))
#     # print('{} working on {}'.format(threading.current_thread().name, [param.grad.data for param in m.parameters()]))
#     tmp = []
#     for i in range(2500):
#         m, o, l = step(x, y, m, o)
#         tmp.append(l)
#     return m, name, tmp


def run_single(x, y, m, o, new_gradient):
    name = multiprocessing.current_process().name
    m, o, l = step(x, y, m, o, new_gradient)
    g = Variable([param.grad.data for param in m.parameters()][0])
    return m, name, l, g


def gradients_sum(gradients):
    # print('gradients: {}'.format(gradients))
    s = torch.stack(gradients)
    su = torch.sum(s, dim=0)
    return su


def run_on_queue(parts_q, model, gradients_q, losses_q, new_gradient):
    o = torch.optim.Adam(model.parameters(), lr=1e-2)
    x, y = parts_q.get()
    m, name, l, g = run_single(x, y, model, o, new_gradient)
    # gradients_q.put(g)
    losses_q.put(l)
    parts_q.put((x, y))
    # print('rq', multiprocessing.current_process().name, g)
    # evt.wait()

def run_on_queue2(parts_q, model):
    o = torch.optim.Adam(model.parameters(), lr=1e-2)
    x, y = parts_q
    new_gradient = None
    m, name, l, g = run_single(x, y, model, o, new_gradient)
    return l
    # gradients_q.put(g)
    # losses_q.put(l)
    # parts_q.put((x, y))
    # print('rq', multiprocessing.current_process().name, g)
    # evt.wait()

def run_monolithic(x, y, w, dx, dy, NUM_ITERATIONS):
    m, o = instantiate_model(dx, dy)
    t0 = time.time()
    # monolithic run
    losses_mono = []
    for i in range(NUM_ITERATIONS):
        m, o, l = step(x, y, m, o)
        losses_mono.append(l)
    # fig, ax = plt.subplots()
    # ax.plot(losses_mono)
    # plt.ylim(0, 10000)
    # plt.xlim(0, NUM_ITERATIONS)
    # fig.savefig('./plots/monolithic.png')
    t1 = time.time()
    # print('monolithic run done in {} msec'.format("%.2f" % (1000 * (t1 - t0))))
    return losses_mono


if __name__ == "__main__":
    import time
    # from multiprocessing import Queue
    import torch.multiprocessing as mp
    from functools import partial

    NUM_ITERATIONS = 1500
    NUM_PARTITIONS = 2
    N = 6  # 50 - 500 - 1000 - 5000
    dx = 1  # log fino a 1M (0-6)
    dy = 5

    x, y, w = init_data(N, dx, dy)
    l_mono = run_monolithic(x, y, w, dx, dy, NUM_ITERATIONS)


    mp.set_start_method('spawn')
    m, o = instantiate_model(dx, dy)

    parts_x = list(torch.split(x, int(x.size()[0] / NUM_PARTITIONS)))
    parts_y = list(torch.split(y, int(x.size()[0] / NUM_PARTITIONS), 1))
    # print('x:', x)
    # print('y:', y)
    #
    # print('parts_x', parts_x)
    # print('parts_y', parts_y)
    m.share_memory()
    mngr = mp.Manager()

    parts = [(i, j) for i, j in zip(parts_x, parts_y)]
    # print('parts:', parts)
    parts_q = mngr.Queue()
    for p in parts:
        parts_q.put(p)

    num_processes = NUM_PARTITIONS

    # models_q = mp.Queue()
    gradients_q = mngr.Queue(maxsize=num_processes)
    losses_q = mngr.Queue(maxsize=num_processes)

    # evt = mp.Event()
    new_gradient = None

    pool = mp.Pool(processes=num_processes)
    for i in range(NUM_ITERATIONS):
        losses = []
        p = partial(run_on_queue2, model=m)
        losses = pool.map(p, parts)


    # for i in range(NUM_ITERATIONS):
    #     processes = []
    #     for pid in range(num_processes):
    #         p = mp.Process(target=run_on_queue, args=(parts_q, m, gradients_q, losses_q, new_gradient))
    #         p.start()
    #         processes.append(p)
    #
    #     for p in processes:
    #         p.join()
    #
    #     local_gradients = []
    #     while not gradients_q.empty():
    #         # print("Got:", gradients_q.get())
    #         local_gradients.append(gradients_q.get())
    #
    #     new_gradient = gradients_sum(local_gradients)
    #     # print('new:', new_gradient)
    #
    #     local_losses = []
    #     while not losses_q.empty():
    #         got = losses_q.get()
    #         # print('got', got)
    #         local_losses.append(got)
    #
    #     #print(i, local_losses)
    #     print(i, sum(local_losses)[0], l_mono[i][0])

    exit()





    fig, axes = plt.subplots(5, 2, sharex=True, sharey=True)
    i, j = 0, 0
    for name, lst in losses.items():
        axes[i % 5, j % 2].plot(lst, label='{}'.format(name))
        legend = axes[i % 5, j % 2].legend(loc='upper right')
        plt.legend()
        plt.ylim(0, 3000)
        plt.xlim(0, 500)
        i += 1
        j += 1
    fig.savefig('./plots/process_loss.png')
    # for k, v in losses.items():
    #     print(k, v[-2:])


