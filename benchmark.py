#!/usr/bin/env python3

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import multi
import time
import csv
import torch
from torch.autograd import Variable
import torch.multiprocessing as mp
import logging


n_iterations = 2500

fieldnames = ['dx', 'dy', 'tmono', 'emono', 'tmulti', 'emulti', 'mse', 'nsplits']

logger = logging.getLogger(__name__)


def run_mono(x, y, model, optimizer):
    t0 = time.time()
    losses, mono_param_l, m = multi.monolithic_run(x, y, model, optimizer, n_iterations)
    westimated = Variable([p.data for p in m.parameters()][0])
    t1 = time.time()
    tmono = round((t1 - t0) * 1000, 2)
    return losses, tmono, westimated, mono_param_l


def plot_losses(mono_losses, multi_losses, N, dx, dy, n_splits, fname):
    fig, ax = plt.subplots()
    ax.plot(mono_losses, label='monolithic')
    if multi_losses is not None:
        ax.plot(multi_losses, color='red', label='multiprocessing')
    plt.title('X:({},{}), W:({},{}), n-splits: {}'.format(N, dx, dx, dy, n_splits))
    plt.legend()
    plt.savefig(fname)
    # plt.show()
    plt.close()


def do_split(x, y, n_splits):
    x_slices = list(torch.split(x, int(x.size()[0] / n_splits)))
    y_slices = list(torch.split(y, int(y.size()[1] / n_splits), dim=1))
    return x_slices, y_slices

if __name__ == "__main__":
    mp.set_start_method('spawn')

    for N in [50, 500]:  # , 500, 1000, 5000]:
        logger.debug('N = {}'.format(N))
        fname = './results/{}.csv'.format(N)
        res = []

        for dx in map(int, np.power(10, range(0, 4))):
            for dy in map(int, np.power(10, range(0, 4))):
                logger.debug('dx {}, dy {}'.format(dx, dy))
                x, y, w = multi.init_data(N, dx, dy)
                logger.info('Starting x: {}, y: {}, w: {}'.format(x.size(), y.size(), w.size()))
                mono_model, mono_optimizer = multi.instantiate_model(dx, dy)
                mono_losses, tmono, west, mono_param_l = run_mono(x, y, mono_model, mono_optimizer)
                emono = torch.sum((west.mm(x.t()) - y) ** 2).data.numpy()[0]
                mono_params = Variable(mono_param_l[0])

                for num_splits in [2, 5, 10]:
                    plotfname = './results/N{}-dx{}-dy{}-s{}.png'.format(N, dx, dy, num_splits)
                    multi_model, multi_optimizer = multi.instantiate_model(dx, dy)
                    x_slices, y_slices = do_split(x, y, num_splits)

                    parts = [(i, j) for i, j in zip(x_slices, y_slices)]

                    multi_model.share_memory()
                    mngr = mp.Manager()

                    q = mngr.Queue(maxsize=num_splits)
                    r = mngr.Queue(maxsize=num_splits)
                    for el in parts:
                        q.put(el)
                        time.sleep(0.1)

                    processes = []
                    multi_losses = []

                    t0 = time.time()
                    for rank in range(num_splits):
                        t = time.time()
                        p = mp.Process(target=multi.run_mini_batch, args=(q, r, multi_model, multi_optimizer,
                                                                          n_iterations))
                        p.start()
                        processes.append(p)
                        p.join()

                    # for p in processes:
                    #     p.join()

                    results = []
                    while not r.empty():
                        results.append(r.get())

                    gradient = multi.torch_list_sum([x['g'] for x in results])
                    multi_losses = np.sum([x['loss'] for x in results], axis=0)

                    t1 = time.time()
                    tmulti = (t1 - t0) * 1000

                    plot_losses(mono_losses, multi_losses, N, dx, dy, num_splits, plotfname)

                    models_dicts = [Variable(d['model']['linear.weight']) for d in results]
                    differences = []
                    estimated_params = []
                    for pos, estimated_param in enumerate(models_dicts):
                        diff = torch.sum((estimated_param - mono_params) ** 2).data.numpy()[0]
                        differences.append(diff)
                        estimated_params.append(estimated_param)

                    mse = np.mean(differences)

                    mean_params = torch.mean(torch.stack(estimated_params, 0), dim=0)

                    emulti = torch.sum((mean_params.mm(x.t()) - y) ** 2).data.numpy()[0]

                    run_result = {'dx': dx, 'dy': dy, 'tmono': tmono, 'emono': emono, 'tmulti': tmulti,
                                  'emulti': emulti, 'mse': mse, 'nsplits': num_splits}
                    res.append(run_result)
                    logger.info(run_result)

            with open(fname, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for dict in res:
                    writer.writerow(dict)

