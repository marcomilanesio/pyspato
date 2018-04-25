#!/usr/bin/env python3

import numpy as np
import multi
import time
import csv
import torch
from torch.autograd import Variable

n_iterations = 2500

fieldnames = ['dx', 'dy', 'msec', 'error']


def main(N, dx, dy):
    t0 = time.time()
    x, y, w = multi.init_data(N, dx, dy)
    model, optimizer = multi.instantiate_model(dx, dy)
    losses, mono_params, m = multi.monolithic_run(x, y, model, optimizer, n_iterations)

    westimated = Variable([p.data for p in m.parameters()][0])
    err = torch.sum((westimated.mm(x.t()) - y) ** 2).data.numpy()[0]

    t1 = time.time()
    tmono = round((t1 - t0) * 1000, 2)


    multi.plot_losses(losses, None, N, dx, dy, 0)

    return tmono, err

if __name__ == "__main__":
    for N in [50]:  # , 500, 1000, 5000]:
        fname = './results/{}.csv'.format(N)
        res = []

        for dx in map(int, np.power(10, range(0, 2))):
            for dy in map(int, np.power(10, range(0, 2))):
                plotfname = './results/{}-{}-{}.png'.format(N, dx, dy)
                t, err = main(N, dx, dy)

                res.append({'dx': dx, 'dy': dy, 'msec': t, 'error': err})
                print(N, dx, dy, t, err)

        with open(fname, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for dict in res:
                writer.writerow(dict)


