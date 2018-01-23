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


def instantiate_model(x):
    model = linmodel.LinModel(1, 5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    prediction = model(x)

    return model, optimizer, prediction


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
    parts = list(torch.split(x, int(x.size()[0] / num_partitions)))
    rdd_x = sc.parallelize(parts).repartition(num_partitions)
    rdd_y = sc.parallelize(y).repartition(num_partitions)
    model = step(rdd_x, rdd_y)
    exit(0)
    return model, W

if __name__ == '__main__':
    sc = SparkContext(appName='pyspato')
    model, W = main(sc)
    print([param.data for param in model.parameters()])
    print(W)
