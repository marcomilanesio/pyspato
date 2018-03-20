import numpy as np

y = np.random.randn(100).reshape(100,1)
x = np.random.randn(100*10).reshape((100,10))
w = np.random.randn(10).reshape([10,1])
print(y.shape, x.shape, w.shape)

E = np.sum((y - x.dot(w))**2)
print(E)
grad = -2 * x.T.dot(y - x.dot(w))
print(grad)

y1 = y[50:,]
y2 = y[:50,]
x1 = x[50:,:]
x2 = x[:50,:]

#print(y1.shape, x1.shape, w.shape)
grad1 = -2 * x1.T.dot(y1 - x1.dot(w))
grad2 = -2 * x2.T.dot(y2 - x2.dot(w))
print(grad1)
print(grad2)
print(all((grad1 + grad2) - grad) < 1e-12)
