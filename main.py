import numpy as np
import torch.optim
from torch.autograd import Variable

from models import linmodel
import utils

from pyspark import SparkContext

NUM_ITERATIONS = 600

N = 1000   # 50 - 500 - 1000 - 5000
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


def save_model_state(model):
    model_dict = model.state_dict()
    try:
        gradients = [param.grad.data for param in model.parameters()]
    except AttributeError:
        gradients = None
    state = {'model': model_dict,
             'gradients': gradients}
    return state


def create_model():
    return linmodel.LinModel(dx, dy)


def torch_step(state):

    def _torch_step(tup):
        x = tup[0]
        y = tup[1]
        m = create_model()
        m.load_state_dict(state['model'])
        m.eval()
        o = torch.optim.Adam(m.parameters(), lr=1e-2)
        gradients = state['gradients']
        print(gradients)
        # if gradients is not None:
        #     print(gradients)
        #     m.linear.weight.register_hook(lambda grad: gradients)

        prediction = m(x)
        loss = linmodel.cost(y, prediction)
        o.zero_grad()
        loss.backward()
        o.step()
        new_state = save_model_state(m)
        return new_state

    return _torch_step


def gradients_sum(gradients):
    # print('gradients: {}'.format(gradients))
    s = torch.stack(gradients)
    su = torch.sum(s, dim=0)
    return su


def run_local(x, y, W):
    m = create_model()
    o = torch.optim.Adam(m.parameters(), lr=1e-2)
    for i in range(NUM_ITERATIONS):
        m, o = local_step(x, y, m, o)

    res = {'param.data': [param.data for param in m.parameters()],
           'W': W}
    return res


def local_step(x, y, model, optimizer):
    prediction = model(x)  # x = (400, 1): x1 = (200. 1). x2 = (200, 1)
    loss = linmodel.cost(y, prediction)
    optimizer.zero_grad()
    loss.backward()  # get the gradients
    optimizer.step()  #
    return model, optimizer


def main(sc, num_partitions=10):
    x, y, W = prepare_input()
    local_result = run_local(x, y, W)
    for k, v in local_result.items():
        print(k, v)

    parts_x = list(torch.split(x, int(x.size()[0] / num_partitions)))
    parts_y = list(torch.split(y, int(x.size()[0] / num_partitions), 1))

    model = create_model()
    state = save_model_state(model)

    rdd_x = sc.parallelize(parts_x).repartition(num_partitions)
    rdd_y = sc.parallelize(parts_y).repartition(num_partitions)

    parts = rdd_x.zip(rdd_y)  # [((100x1), (5x100)), ...]

    for i in range(NUM_ITERATIONS):
        print('.', end='')
        intermediate = (parts.map(torch_step(state))
                        .map(lambda j: (j['model'], j['gradients'][0]))
                        .collect()
                        )
        models = [j[0] for j in intermediate]
        gradients = [j[1] for j in intermediate]

        state = {'model': models[0],
                 'gradients': Variable(gradients_sum(gradients))}

        # if all(torch.equal(x['linear.weight'], models[0]['linear.weight']) for x in models):
        #     state = {'model': models[0],
        #              'gradients': Variable(gradients_sum(gradients))}
        # else:
        #     state = {'model': models[0],
        #              'gradients': Variable(gradients_sum(gradients))}

        # print(gradients)
        # print(models[0])

    final_model = linmodel.LinModel(dx, dy)
    final_model.load_state_dict(models[0])
    final_model.eval()
    print('\n')
    print([param.data for param in final_model.parameters()])
    return None, W


if __name__ == '__main__':
    sc = SparkContext(appName='pyspato')
    model, W = main(sc)
    # print([param.data for param in model.parameters()])
    # print(W)
