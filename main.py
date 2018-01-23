import numpy as np
import torch.optim
from torch.autograd import Variable

from models import linmodel
import utils

Nsamples = 400  # TO SCALE
Xold = np.linspace(0, 1000, Nsamples).reshape([Nsamples, 1])
X = utils.standardize(Xold)

W = np.random.randint(1, 10, size=(5, 1))

Y = W.dot(X.T)  # target

for i in range(Y.shape[1]):
    Y[:, i] = utils.add_noise(Y[:, i])

x = Variable(torch.from_numpy(X), requires_grad=False).type(torch.FloatTensor)
y = Variable(torch.from_numpy(Y), requires_grad=False).type(torch.FloatTensor)

model = linmodel.LinModel(1, 5)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

for i in range(5000):
    prediction = model(x)  # x = (400, 1): x1 = (200. 1). x2 = (200, 1)
    loss = linmodel.cost(y, prediction)
    optimizer.zero_grad()
    loss.backward()  # get the gradients
    assert not any([x.grad, y.grad, loss.grad])
    # sum gradients

    optimizer.step()  #



print([param.data for param in model.parameters()])
print(W)
