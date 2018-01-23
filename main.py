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
    return x, y, W


def instantiate_model(tup):
    model = linmodel.LinModel(1, 5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    full = (*tup, model, optimizer)
    return full


def step(rdd1, rdd2):
    rdd = rdd1.map(instantiate_model)
    print(rdd.take(1))
    #print(rdd2.count())
    return None

    model = linmodel.LinModel(1, 5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    for i in range(5000):
        prediction = model(v1)  # x = (400, 1): x1 = (200. 1). x2 = (200, 1)
        loss = linmodel.cost(v2, prediction)
        optimizer.zero_grad()
        loss.backward()  # get the gradients

        # sum gradients

        optimizer.step()  #
    return model


def main(sc, num_partitions=4):
    x, y, W = prepare_input()
    parts_x = list(torch.split(x, int(x.size()[0] / num_partitions)))
    parts_y = list(torch.split(y, int(x.size()[0] / num_partitions), 1))

    rdd_x = sc.parallelize(parts_x).repartition(num_partitions)
    rdd_y = sc.parallelize(parts_y).repartition(num_partitions)

    parts = (rdd_x.zip(rdd_y)  # [((100x1), (5x100)), ...]
             .map(instantiate_model)  # [((100,1), (5,100), m1, o1), ... )
             .cache())

    print([type(x) for x in parts.take(1)[0]])

    exit(0)
    return model, W

if __name__ == '__main__':
    sc = SparkContext(appName='pyspato')
    model, W = main(sc)
    print([param.data for param in model.parameters()])
    print(W)
