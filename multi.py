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


def gradients_sum(gradients):
    # print('gradients: {}'.format(gradients))
    s = torch.stack(gradients)
    su = torch.sum(s, dim=0)
    return su


# def run_on_queue(parts_q, model, gradients_q, losses_q, new_gradient):
#     o = torch.optim.Adam(model.parameters(), lr=1e-2)
#     x, y = parts_q.get()
#     m, name, l, g = run_single(x, y, model, o, new_gradient)
#     # gradients_q.put(g)
#     losses_q.put(l)
#     parts_q.put((x, y))
#     # print('rq', multiprocessing.current_process().name, g)
#     # evt.wait()

# def run_on_queue2(parts_q, model):
#     o = torch.optim.Adam(model.parameters(), lr=1e-2)
#     x, y = parts_q
#     new_gradient = None
#     m, name, l, g = run_single(x, y, model, o, new_gradient)
#     return l
#     # gradients_q.put(g)
#     # losses_q.put(l)
#     # parts_q.put((x, y))
#     # print('rq', multiprocessing.current_process().name, g)
#     # evt.wait()


# def run(parts, model, optimizer, queue_g, new_g):
#     name = multiprocessing.current_process().name
#     x, y = parts
#     m, o, l = step(x, y, model, optimizer, new_g)
#     g = Variable([param.grad.data for param in m.parameters()][0])
#     w_temp = [param.data for param in m.parameters()][0]
#     close_grad = grad_closed_form(x, y, w_temp)
#     print('in run method of {}'.format(name))
#     print(close_grad)
#     print(g)
#     print('--')
#     queue_g.put(g)
#     return x.size()


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
    print('monolithic run done in {} msec'.format("%.2f" % (1000 * (t1 - t0))))
    estimated = [param.data for param in model.parameters()]
    return losses, estimated, model

if __name__ == "__main__":

    NUM_ITERATIONS = 2500
    NUM_PARTITIONS = 2
    N = 500  # 50 - 500 - 1000 - 5000
    dx = 10  # log fino a 1M (0-6)
    dy = 1

    # torch variables
    x, y, w = init_data(N, dx, dy)
    model, optimizer = instantiate_model(dx, dy)
    losses, mono_params, m = monolithic_run(x, y, model, optimizer, NUM_ITERATIONS)
    plot_loss(losses, False)


    print(w)
    # print([param.data for param in m.parameters()])
    print(m.state_dict())
    exit()


    mp.set_start_method('spawn')
    model, optimizer = instantiate_model(dx, dy)

    parts_x = list(torch.split(x, int(x.size()[0] / NUM_PARTITIONS)))
    parts_y = list(torch.split(y, int(y.size()[0] / NUM_PARTITIONS)))
    parts = [(i, j) for i, j in zip(parts_x, parts_y)]
    print('number of splits = {}'.format(len(parts)))
    model.share_memory()
    mngr = mp.Manager()

    num_processes = NUM_PARTITIONS

    gradients_q = mngr.Queue(maxsize=num_processes)

    # evt = mp.Event()
    new_gradient = None

    with mp.Pool(processes=num_processes) as pool:
        for i in range(NUM_ITERATIONS):
            p = partial(run, model=model, optimizer=optimizer, queue_g=gradients_q, new_g=new_gradient)
            res = pool.map(p, parts)
            local_gradients = []
            while not gradients_q.empty():
                local_gradients.append(gradients_q.get())

            new_gradient = gradients_sum(local_gradients)
            print('pool', new_gradient)


        # for i in range(NUM_ITERATIONS):
        #     losses = []
        #     p = partial(run_on_queue2, model=model)
        #     losses = pool.map(p, parts)
    exit()

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


