import numpy as np
import torch.optim
from torch.autograd import Variable

from models import linmodel
import utils

from pyspark import SparkContext


def prepare_input(nsamples=400):
    Xold = np.linspace(0, 1000, nsamples).reshape([nsamples, 1])
    X = utils.standardize(Xold)

    W = np.random.randint(1, 10, size=(5, 1))

    Y = W.dot(X.T)  # target

    for i in range(Y.shape[1]):
        Y[:, i] = utils.add_noise(Y[:, i])

    x = Variable(torch.from_numpy(X), requires_grad=False).type(torch.FloatTensor)
    y = Variable(torch.from_numpy(Y), requires_grad=False).type(torch.FloatTensor)
    print("created torch variables {} {}".format(x.size(), y.size()))
    return x, y, W


def save_state(model, optimizer):
    model_dict = model.state_dict()
    optimizer_dict = optimizer.state_dict()
    gradients = [param.grad.data for param in model.parameters()]
    state = {'model': model_dict,
             'optimizer': optimizer_dict,
             'gradients': gradients}
    return state


def initialize(tup):
    x, y = tup[0]   # data
    m, o = tup[1]   # models and optimizer
    model, optimizer = torch_step(x, y, m, o)
    state = save_state(model, optimizer)
    return x, y, state


def create_model():
    model = linmodel.LinModel(1, 5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    return model, optimizer


def torch_step(x, y, model, optimizer):
    prediction = model(x)
    loss = linmodel.cost(y, prediction)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return model, optimizer


# def step(tup, gradient=None):
#     x = tup[0]
#     y = tup[1]
#     model = tup[2]
#     optimizer = tup[3]
#     prediction = model(x)
#     loss = linmodel.cost(y, prediction)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
#     # gradient = [param.grad.data for param in model.parameters()]
#
#     return x, y, model, optimizer
#     """
#     model = linmodel.LinModel(1, 5)
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
#
#     for i in range(5000):
#         prediction = model(v1)  # x = (400, 1): x1 = (200. 1). x2 = (200, 1)
#         loss = linmodel.cost(v2, prediction)
#         optimizer.zero_grad()
#         loss.backward()  # get the gradients
#
#         # sum gradients
#
#         optimizer.step()  #
#     return model
#     """


def gradients_sum(gradients):
    # print('gradients: {}'.format(gradients))
    s = torch.stack(gradients)
    su = torch.sum(s, dim=0)
    return su


def restore_model(tup):
    x = tup[0]
    y = tup[1]
    d = tup[2]
    model, optimizer = create_model()
    model.load_state_dict(d['model'])
    optimizer.load_state_dict(d['optimizer'])
    model.eval()

    model, optimizer = torch_step(x, y, model, optimizer)
    # return [p for p in model.parameters()]
    state = save_state(model, optimizer)
    return x, y, state


def run_loops(tup):
    x = tup[0]
    y = tup[1]
    d = tup[2]
    model, optimizer = create_model()
    model.load_state_dict(d['model'])
    optimizer.load_state_dict(d['optimizer'])
    model.eval()
    for i in range(1000):
        model, optimizer = torch_step(x, y, model, optimizer)
    return model

# def restore_model(new_gradient):
#
#     def _restore_model(tup):
#         x = tup[0]
#         y = tup[1]
#         d = tup[2]
#         model, optimizer = create_model()
#         model.load_state_dict(d['model'])
#         optimizer.load_state_dict(d['optimizer'])
#         model.eval()
#         print("_restore_model inner: ", [param.data for param in model.parameters()])
#         # model.linear.weight.register_hook(lambda grad: new_gradient)
#
#         model, optimizer = torch_step(x, y, model, optimizer)
#         # return [p for p in model.parameters()]
#         state = save_state(model, optimizer)
#         return x, y, state
#
#     return _restore_model


def main(sc, num_partitions=4):
    x, y, W = prepare_input()
    parts_x = list(torch.split(x, int(x.size()[0] / num_partitions)))
    parts_y = list(torch.split(y, int(x.size()[0] / num_partitions), 1))

    rdd_models = sc.parallelize([create_model() for _ in range(num_partitions)]).repartition(num_partitions)

    rdd_x = sc.parallelize(parts_x).repartition(num_partitions)
    rdd_y = sc.parallelize(parts_y).repartition(num_partitions)

    parts = rdd_x.zip(rdd_y)  # [((100x1), (5x100)), ...]

    # <'torch.autograd.variable.Variable'>, < class 'torch.autograd.variable.Variable' >, < class 'dict' >
    full = parts.zip(rdd_models).map(initialize).cache()

    # gradients = full.map(lambda j: j[2]['gradients'][0]).collect()
    # new_gradient = Variable(gradients_sum(gradients))

    # print('round 0')
    # print(type(new_gradient), new_gradient.size())
    # print(new_gradient)

    full = full.map(run_loops)

    # for i in range(100):
    #     full = full.map(restore_model)
    #     # full = full.map(restore_model(new_gradient))
    #     # grads = full.map(lambda x: x[2]['gradients'][0]).collect()
    #     # new_gradient = Variable(gradients_sum(grads))
    #     # print('round {}'.format(i))
    #     # print(type(new_g), new_g.size())
    #     # print(new_g)

    result = full.collect()
    for m in result:
        print([param.data for param in m.parameters()])
    print('target: \n{}'.format(W))

    return None, W

if __name__ == '__main__':
    sc = SparkContext(appName='pyspato')
    model, W = main(sc)
    # print([param.data for param in model.parameters()])
    # print(W)
