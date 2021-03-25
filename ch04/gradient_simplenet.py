import numpy as np
from common.activation_function import softmax
from common.error_function import cross_entropy_error
from common.differential_function import numerical_gradient


class SimpleNet:
    def __init__(self):
        self.W = np.array([[0.47355232, 0.9977393, 0.84668094],
                           [0.85557411, 0.03563661, 0.69422093]])

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        return cross_entropy_error(y, t)


if __name__ == '__main__':
    net = SimpleNet()
    print(net.W)
    x = np.array([0.6, 0.9])
    p = net.predict(x)
    print(p)
    print(np.argmax(p))
    t = np.array([0, 0, 1])
    print(net.loss(x, t))

    def f(W):
        return net.loss(x, t)

    dW = numerical_gradient(f, net.W)
    print(dW)
