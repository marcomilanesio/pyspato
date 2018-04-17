import numpy as np
import multi
import time
import csv

n_iterations = 2500

fieldnames = ['dx', 'dy', 'msec', 'error']

def main(N, dx, dy):
    t0 = time.time()
    x, y, w = multi.init_data(N, dx, dy)
    model, optimizer = multi.instantiate_model(dx, dy)
    losses, mono_params, m = multi.monolithic_run(x, y, model, optimizer, n_iterations)
    westimated = [p.data for p in m.parameters()][0]
    t1 = time.time()
    tres = (t1 - t0) * 1000
    return tres, westimated.numpy(), x.data.numpy(), y.data.numpy()

if __name__ == "__main__":
    for N in [50, 500, 1000, 5000]:
        fname = '{}.csv'.format(N)
        res = []

        for dx in map(int, np.power(10, range(0, 7))):
            for dy in map(int, np.power(10, range(0, 7))):
                t, we, x, y = main(N, dx, dy)
                err = np.sum((we.dot(x.T) - y) ** 2)
                res.append({'dx': dx, 'dy': dy, 'msec': t, 'error': err})
                print(N, dx, dy, t, err)

        with open(fname, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for dict in res:
                writer.writerow(dict)


