import numpy as np
import torch.optim
from torch.autograd import Variable

from models import linmodel
import utils
import threading

NUM_ITERATIONS = 1500

N = 400   # 50 - 500 - 1000 - 5000
dx = 1   # log fino a 1M (0-6)
dy = 5


def prepare_input(nsamples=N):
    Xold = np.linspace(0, 1000, nsamples * dx).reshape([nsamples, dx])
    X = utils.standardize(Xold)

    W = np.random.randint(1, 10, size=(dy, dx))

    Y = W.dot(X.T)  # target

    for i in range(Y.shape[1]):
        Y[:, i] = utils.add_noise(Y[:, i])

    x = Variable(torch.from_numpy(X), requires_grad=False).type(torch.FloatTensor)
    y = Variable(torch.from_numpy(Y), requires_grad=False).type(torch.FloatTensor)
    print("created torch variables {} {}".format(x.size(), y.size()))
    return x, y, W


# def save_model_state(model):
#     model_dict = model.state_dict()
#     try:
#         gradients = [param.grad.data for param in model.parameters()]
#     except AttributeError:
#         gradients = None
#     state = {'model': model_dict,
#              'gradients': gradients}
#     return state


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


def run(model, x, y):
    model = local_step(x, y, model)
    local_grad = Variable([param.grad.data for param in model.parameters()][0])
    return model, local_grad


def main(num_partitions=4):
    x, y, W = prepare_input()

    parts_x = list(torch.split(x, int(x.size()[0] / num_partitions)))
    parts_y = list(torch.split(y, int(x.size()[0] / num_partitions), 1))

    q = [(i, j) for i, j in zip(parts_x, parts_y)]
    model = create_model(dx, dy)

    x1, y1 = q[0]
    x2, y2 = q[1]
    x3, y3 = q[2]
    x4, y4 = q[3]
    for _ in range(250):
        tmp = []
        model1, t1 = run(model, x1, y1)
        tmp.append(t1)

        model2, t2 = run(model, x2, y2)
        tmp.append(t2)

        model3, t3 = run(model, x3, y3)
        tmp.append(t3)

        model4, t4 = run(model, x4, y4)
        tmp.append(t4)

        s = gradients_sum(tmp)
        model = update_model(model1, s)

    return model, W


if __name__ == '__main__':
    model, W = main()
    print([param.data for param in model.parameters()])
    print(W)
