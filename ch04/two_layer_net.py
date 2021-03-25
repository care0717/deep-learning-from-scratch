import numpy as np
from common.activation_function import sigmoid, softmax
from common.error_function import cross_entropy_error
from common.differential_function import numerical_gradient


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {'W1': weight_init_std * np.random.randn(input_size, hidden_size), 'b1': np.zeros(hidden_size),
                       'W2': weight_init_std * np.random.randn(hidden_size, output_size), 'b2': np.zeros(output_size)}

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2

        return softmax(a2)

    def loss(self, x, t):
        y = self.predict(x)

        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        return np.sum(y == t) / float(x.shape[0])

    def numerical_gradient(self, x, t):
        def loss_w(w): return self.loss(x, t)
        return {'W1': numerical_gradient(loss_w, self.params['W1']), 'b1': numerical_gradient(loss_w, self.params['b1']),
                'W2': numerical_gradient(loss_w, self.params['W2']), 'b2': numerical_gradient(loss_w, self.params['b2'])}
