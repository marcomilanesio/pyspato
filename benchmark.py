import numpy as np
import multi
import time

n_iterations = 2500

def main():
    for N in [50, 500, 1000, 5000]:
        for dx in map(int, np.power(10, range(0, 7))):
            for dy in map(int, np.power(10, range(0, 7))):
                t0 = time.time()
                x, y, w = multi.init_data(N, dx, dy)
                model, optimizer = multi.instantiate_model(dx, dy)
                losses, mono_params, m = multi.monolithic_run(x, y, model, optimizer, n_iterations)
                t1 = time.time()
                tres = (t1 - t0) * 1000
                print(N, dx, dy, tres)

if __name__ == "__main__":
    main()
