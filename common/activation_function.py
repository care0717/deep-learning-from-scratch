import numpy as np


def step_function(x):
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def identity_function(x):
    return x


def softmax(x):
    c = np.max(x)
    exp_x = np.exp(x - c)
    sum_exp = np.sum(exp_x)
    return exp_x/sum_exp
