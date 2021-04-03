import numpy as np
from common.differential_function import numerical_gradient
from common.layer import Affine, Relu, SoftmaxWithLoss, BatchNormalization
from collections import OrderedDict


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01, use_batchnorm=False):
        self.params = {'W1': weight_init_std * np.random.randn(input_size, hidden_size), 'b1': np.zeros(hidden_size),
                       'W2': weight_init_std * np.random.randn(hidden_size, output_size), 'b2': np.zeros(output_size)}

        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        if use_batchnorm:
            self.params['gamma1'] = np.ones(hidden_size)
            self.params['beta1'] = np.zeros(hidden_size)
            self.layers['BatchNorm1'] = BatchNormalization(
                gamma=self.params['gamma1'], beta=self.params['beta1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.use_batchnorm = use_batchnorm

        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)

        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        return np.sum(y == t) / float(x.shape[0])

    def numerical_gradient(self, x, t):
        def loss_w(w): return self.loss(x, t)
        return {'W1': numerical_gradient(loss_w, self.params['W1']), 'b1': numerical_gradient(loss_w, self.params['b1']),
                'W2': numerical_gradient(loss_w, self.params['W2']), 'b2': numerical_gradient(loss_w, self.params['b2'])}

    def gradient(self, x, t):
        self.loss(x, t)
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        dparam = {
            'W1': self.layers['Affine1'].dW, 'b1': self.layers['Affine1'].db,
            'W2': self.layers['Affine2'].dW, 'b2': self.layers['Affine2'].db,
        }
        if self.use_batchnorm:
            dparam['beta1'] = self.layers['BatchNorm1'].dbeta
            dparam['gamma1'] = self.layers['BatchNorm1'].dgamma
        return dparam
