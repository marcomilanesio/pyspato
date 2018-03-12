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


def step(x, y, model, optimizer):
    prediction = model(x)  # x = (400, 1): x1 = (200. 1). x2 = (200, 1)
    loss = linmodel.cost(y, prediction)
    optimizer.zero_grad()
    loss.backward()  # get the gradients
    # print([param.grad.data for param in model.parameters()])
    # sum gradients
    optimizer.step()  #
    return model, optimizer, loss.data.numpy()


def run(x, y, m, o):
    name = threading.current_thread().name
    print('{} working on {}-{}'.format(name, x.size(), y.size()))
    # print('{} working on {}'.format(threading.current_thread().name, [param.grad.data for param in m.parameters()]))
    tmp = []
    for i in range(2500):
        m, o, l = step(x, y, m, o)
        tmp.append(l)
    return m, name, tmp


def run_single(x, y, m, o):
    name = multiprocessing.current_process().name
    m, o, l = step(x, y, m, o)
    g = Variable([param.grad.data for param in m.parameters()][0])
    return m, name, l, g


def gradients_sum(gradients):
    # print('gradients: {}'.format(gradients))
    s = torch.stack(gradients)
    su = torch.sum(s, dim=0)
    return su


def run_on_queue(parts_q, model, gradients_q):
    o = torch.optim.Adam(model.parameters(), lr=1e-2)
    x, y = parts_q.get()
    m, name, l, g = run_single(x, y, model, o)
    gradients_q.put(g)
    print(multiprocessing.current_process().name, x.size(), y.size(), g, gradients_q.qsize(), '\n')
    # evt.wait()
    return g


def run_monolithic(x, y, w, dx, dy, NUM_ITERATIONS):
    m, o = instantiate_model(dx, dy)
    t0 = time.time()
    # monolithic run
    losses_mono = []
    for i in range(NUM_ITERATIONS):
        m, o, l = step(x, y, m, o)
        losses_mono.append(l)
    fig, ax = plt.subplots()
    ax.plot(losses_mono)
    plt.ylim(0, 10000)
    plt.xlim(0, NUM_ITERATIONS)
    fig.savefig('./plots/monolithic.png')
    t1 = time.time()
    print('monolithic run done in {} msec'.format("%.2f" % (1000 * (t1 - t0))))



if __name__ == "__main__":
    import time
    # from multiprocessing import Queue
    import torch.multiprocessing as mp

    NUM_ITERATIONS = 5000
    NUM_PARTITIONS = 10
    N = 50  # 50 - 500 - 1000 - 5000
    dx = 1  # log fino a 1M (0-6)
    dy = 5

    x, y, w = init_data(N, dx, dy)
    # run_monolithic(x, y, w, dx, dy, NUM_ITERATIONS)

    mp.set_start_method('spawn')
    m, o = instantiate_model(dx, dy)

    parts_x = list(torch.split(x, int(x.size()[0] / NUM_PARTITIONS)))
    parts_y = list(torch.split(y, int(x.size()[0] / NUM_PARTITIONS), 1))

    m.share_memory()
    mngr = mp.Manager()

    parts = [(i, j) for i, j in zip(parts_x, parts_y)]
    parts_q = mngr.Queue()
    for p in parts:
        parts_q.put(p)

    # models_q = mp.Queue()
    gradients_q = mngr.Queue(maxsize=10)

    num_processes = 10
    processes = []
    # evt = mp.Event()

    for pid in range(num_processes):
        p = mp.Process(target=run_on_queue, args=(parts_q, m, gradients_q))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    local_gradients = []
    while not gradients_q.empty():
        # print("Got:", gradients_q.get())
        local_gradients.append(gradients_q.get())

    new_gradient = gradients_sum(local_gradients)
    print('new:', new_gradient)
    exit()



    losses = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        for i in range(NUM_ITERATIONS):
            models = []
            gradients = []
            jobs = {executor.submit(run_single, t[0], t[1], m, o): t for t in q}
            for future in concurrent.futures.as_completed(jobs):
                res = jobs[future]
                try:
                    mm, name, lst, local_g = future.result()
                    # print(name, [param.grad.data for param in mm.parameters()])
                except Exception as exc:
                    print('ach', res)
                else:
                    try:
                        losses[name].append(lst)
                    except KeyError:
                        losses[name] = [lst]
                    models.append(mm)
                    gradients.append(local_g)
            m = models[0]
            m.linear.weight.grad = gradients_sum(gradients)
            if i % 500 == 0:
                print(i, end='\r')

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

