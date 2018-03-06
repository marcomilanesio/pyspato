import numpy as np
import torch.optim
from torch.autograd import Variable

from models import linmodel
import utils
import numpy.linalg as linalg
import sys


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


def get_mse(m, w):
    r = np.array([param.data for param in m.parameters()])
    res = Variable(r[0])
    return linmodel.mse(res, w)

if __name__ == "__main__":
    NUM_ITERATIONS = 10000

    N = 5000  # 50 - 500 - 1000 - 5000
    dx = 10000  # log fino a 1M (0-6)
    dy = 5

    x, y, w = init_data(N, dx, dy)

    m, o = instantiate_model(dx, dy)

    losses = []
    for i in range(NUM_ITERATIONS):
        m, o, l = step(x, y, m, o)
        losses.append(l)

    print(min(losses), max(losses))
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(losses, label='{}/{}'.format(N, dx))
    legend = ax.legend(loc='upper right')
    plt.ylim(0, 10000)
    plt.xlim(0, NUM_ITERATIONS)
    plt.show()
    # print(get_mse(m, w))
    # print(w)
    # print([param.data for param in m.parameters()])
    pred = m(x)
    plt.scatter(pred.data.numpy(), y.data.numpy())
    plt.show()
