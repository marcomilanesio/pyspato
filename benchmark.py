import numpy as np
import main
import time
import csv

fname = './fixed_n_iterations.csv'
fieldnames = ['n_iterations', 'dx', 'mse', 'time']

n_iterations = 5000
n_samples = [50, 500, 1000, 5000]
dx = np.power(10, range(1, 7))
dy = 5

results = []

for N in [50, 500, 1000, 5000]:
    for dx in [1, 10, 100, 1000, 10000, 100000, 1000000]:
        print("running ", N, dx)
        start = time.time()
        x, y, w = main.init_data(N, dx, dy)
        m, o = main.instantiate_model(dx, dy)
        for i in range(n_iterations):
            m, o = main.step(x, y, m, o)

        results.append((N, dx, main.get_mse(m, w), time.time() - start))

print('Saving results...')
with open(fname, 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(fieldnames)
    for row in results:
        writer.writerow(row)
print('Done')