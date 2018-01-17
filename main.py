import torch.optim
import numpy as np
from torch.autograd import Variable
import linmodel


def standardize(X):
    return 2 * (X - np.min(X)) / (np.max(X) - np.min(X)) - 1, np.min(X), np.max(X)


def add_noise(y):
    rnd = np.random.randn(y.shape[0])
    return y + rnd


Nsamples = 400  # TO SCALE
Xold = np.linspace(0, 1000, Nsamples).reshape([Nsamples, 1])
X = standardize(Xold)[0]

W = np.random.randint(1, 10, size=(5, 1))

Y = W.dot(X.T)  # target

for i in range(Y.shape[1]):
    Y[:, i] = add_noise(Y[:, i])

x = Variable(torch.from_numpy(X), requires_grad=False).type(torch.FloatTensor)
y = Variable(torch.from_numpy(Y), requires_grad=False).type(torch.FloatTensor)

# how to split x, y

model = linmodel.LinModel(1, 5)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

for i in range(5000):
    prediction = model(x)  # x = (400, 1): x1 = (200. 1). x2 = (200, 1)
    loss = linmodel.cost(y, prediction)
    optimizer.zero_grad()
    loss.backward()  # get the gradients

    # sum gradients

    optimizer.step()  #

print([param.data for param in model.parameters()])
print(W)
