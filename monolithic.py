import torch.optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from models import linmodel
import utils
import time


def init_data(nsamples, dx, dy):
    X, Y, W = utils.init_data(nsamples, dx, dy)

    x = Variable(torch.from_numpy(X), requires_grad=True).type(torch.FloatTensor)
    y = Variable(torch.from_numpy(Y), requires_grad=True).type(torch.FloatTensor)
    w = Variable(torch.from_numpy(W), requires_grad=True).type(torch.FloatTensor)
    return x, y, w


def instantiate_model(dx, dy):
    model = linmodel.LinModel(dx, dy)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    return model, optimizer


def torch_list_sum(list_):
    s = torch.stack(list_)
    su = torch.sum(s, dim=0)
    return su


def plot_loss(loss, fname=False):
    fig, ax = plt.subplots()
    ax.plot(loss)
    # plt.ylim(0, 10000)
    # plt.xlim(0, NUM_ITERATIONS)
    if fname:
        fig.savefig(fname)
    else:
        plt.show()
    plt.close()


def run(x, y, model, optimizer, num_iterations):
    losses = []
    for i in range(num_iterations):
        optimizer.zero_grad()
        prediction = model(x)
        loss = linmodel.cost(y, prediction)
        loss.backward()
        optimizer.step()
        losses.append(loss.data.numpy())

    estimated = [param.data for param in model.parameters()]
    return losses, estimated, model


if __name__ == "__main__":

    NUM_ITERATIONS = 2500
    N = 500  # 50 - 500 - 1000 - 5000
    dx = 10  # log fino a 1M (0-6)
    dy = 5

    # torch variables
    x, y, w = init_data(N, dx, dy)
    model, optimizer = instantiate_model(dx, dy)
    t0 = time.time()
    mono_losses, mono_params, m = run(x, y, model, optimizer, NUM_ITERATIONS)
    t1 = time.time()
    t_mono = (t1 - t0) * 1000
    print('monolithic run: {} msec'.format(t_mono))
    print(w)
    print(m.state_dict())
    plot_loss(mono_losses)
