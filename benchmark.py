import numpy as np
import multi
import time
import csv
from functools import partial
import torch.multiprocessing as mp
import torch

fname = './pool.csv'
fieldnames = ['n_iterations', 'n_partitions', 'dx', 'time']

NUM_PARTITIONS = 10
n_iterations = 1500
n_samples = [50, 500, 1000, 5000]
dx = np.power(10, range(1, 7))
dy = 5

results = []


def main():
    for N in [50, 500, 1000, 5000]:
        for npart in [10, 100, 1000]:
            for dx in [1, 10, 100, 1000, 10000, 100000, 100000]:
                print("running ", N, dx)
                start = time.time()
                x, y, w = multi.init_data(N, dx, dy)
                m, o = multi.instantiate_model(dx, dy)
                parts_x = list(torch.split(x, int(x.size()[0] / NUM_PARTITIONS)))
                parts_y = list(torch.split(y, int(x.size()[0] / NUM_PARTITIONS), 1))
                m.share_memory()

                parts = [(i, j) for i, j in zip(parts_x, parts_y)]
                num_processes = NUM_PARTITIONS
                pool = mp.Pool(processes=num_processes)
                for i in range(n_iterations):
                    local_losses = []
                    p = partial(multi.run_on_queue2, model=m)
                    local_losses = pool.map(p, parts)
                print(N, npart, dx, time.time() - start)
                results.append((N, npart, dx, time.time() - start))
                pool.close()
                pool.join()

if __name__ == "__main__":
    mp.set_start_method('spawn')

    main()
    print('Saving results...')
    with open(fname, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(fieldnames)
        for row in results:
            writer.writerow(row)

    print('Done')