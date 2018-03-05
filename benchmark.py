import numpy as np
import main
import time
import csv

fname = './fixed_n_iterations.csv'
fieldnames = ['n_iterations', 'n_partitions', 'dx', 'mse', 'time']

n_iterations = 300
n_samples = [50, 500, 1000, 5000]
dx = np.power(10, range(1, 7))
dy = 5

results = []

for N in [50, 500, 1000, 5000]:
    for npart in [10, 100, 1000]:
        for dx in [1, 10, 100, 1000, 10000, 100000, 1000000]:
            print("running ", N, dx)
            start = time.time()
            model, W = main.main(sample_size=N, dx=dx, dy=dy, num_partitions=npart)
            mse = main.get_mse(model, W)
            results.append((N, npart, dx, mse, time.time() - start))

print('Saving results...')
with open(fname, 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(fieldnames)
    for row in results:
        writer.writerow(row)

print('Done')