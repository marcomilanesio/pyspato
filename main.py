import numpy as np
import torch.optim
from torch.autograd import Variable

from models import linmodel
import utils


def init_data(nsamples=400):
    n = nsamples  # TO SCALE
    Xold = np.linspace(0, 1000, n).reshape([n, 1])
    X = utils.standardize(Xold)

    W = np.random.randint(1, 10, size=(5, 1))

    Y = W.dot(X.T)  # target

    for i in range(Y.shape[1]):
        Y[:, i] = utils.add_noise(Y[:, i])

    x = Variable(torch.from_numpy(X), requires_grad=True).type(torch.FloatTensor)
    y = Variable(torch.from_numpy(Y), requires_grad=True).type(torch.FloatTensor)
    w = Variable(torch.from_numpy(W), requires_grad=True).type(torch.FloatTensor)
    return x, y, w


def instantiate_model(x, y):
    model = linmodel.LinModel(1, 5)
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
    return model, optimizer

if __name__ == "__main__":
    x, y, w = init_data()
    m, o = instantiate_model(x, y)
    for i in range(5000):
        m, o = step(x, y, m, o)

    r = np.array([param.data for param in m.parameters()])
    res = Variable(r[0])

    # print(w.size(), res.size())
    # print(type(w), type(res))
    # print(w, res)
    #
    # m = torch.sum((w - res)**2)
    # print(m)
    print(linmodel.mse(res, w))