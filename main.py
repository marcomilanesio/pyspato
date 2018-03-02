import numpy as np
import torch.optim
from torch.autograd import Variable

from models import linmodel
import utils
# from multiprocessing import Pool, Manager
from functools import partial
from concurrent import futures
import multiprocessing

NUM_ITERATIONS = 1500

N = 400   # 50 - 500 - 1000 - 5000
dx = 1   # log fino a 1M (0-6)
dy = 5

end = -1


def prepare_input(nsamples=N):
    Xold = np.linspace(0, 1000, nsamples * dx).reshape([nsamples, dx])
    X = utils.standardize(Xold)

    W = np.random.randint(1, 10, size=(dy, dx))

    Y = W.dot(X.T)  # target

    for i in range(Y.shape[1]):
        Y[:, i] = utils.add_noise(Y[:, i])

    x = Variable(torch.from_numpy(X), requires_grad=False).type(torch.FloatTensor)
    y = Variable(torch.from_numpy(Y), requires_grad=False).type(torch.FloatTensor)
    w = Variable(torch.from_numpy(W), requires_grad=True).type(torch.FloatTensor)
    # print("created torch variables {} {}".format(x.size(), y.size()))
    return x, y, w


def create_model(insize, outsize):
    return linmodel.LinModel(insize, outsize)


def gradients_sum(gradients):
    # print('gradients: {}'.format(gradients))
    s = torch.stack(gradients)
    su = torch.sum(s, dim=0)
    return su


def local_step(x, y, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    prediction = model(x)  # x = (400, 1): x1 = (200. 1). x2 = (200, 1)
    loss = linmodel.cost(y, prediction)
    optimizer.zero_grad()
    loss.backward()  # get the gradients
    optimizer.step()  #
    return model


def update_model(model, g):
    model.linear.weight.grad = g
    return model


def run(tup, model):
    x, y = tup
    model = local_step(x, y, model)
    local_grad = Variable([param.grad.data for param in model.parameters()][0])
    return model, local_grad


def get_mse(m, w):
    r = np.array([param.data for param in m.parameters()])
    res = Variable(r[0])
    return linmodel.mse(res, w).data.numpy()[0]


def main(num_partitions=4):
    x, y, W = prepare_input()

    parts_x = list(torch.split(x, int(x.size()[0] / num_partitions)))
    parts_y = list(torch.split(y, int(x.size()[0] / num_partitions), 1))

    q = [(i, j) for i, j in zip(parts_x, parts_y)]
    model = create_model(dx, dy)

    with futures.ProcessPoolExecutor(12) as executor:
        for i in range(300):
            jobs = [executor.submit(run, chunk, model) for chunk in q]
            local_gradients = []
            local_models = []
            for comp_job in futures.as_completed(jobs):
                local_gradients.append(comp_job.result()[1])
                local_models.append(comp_job.result()[0])
            new_gradient = gradients_sum(local_gradients)
            model = update_model(local_models[0], new_gradient)

    return model, W


if __name__ == '__main__':
    model, W = main()
    print("mse: ", get_mse(model, W))
    # print([param.data for param in model.parameters()])
    # print(W)
