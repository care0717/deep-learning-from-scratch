from common.differential_function import numerical_gradient
import numpy as np


def function_2(x):
    return x[0]**2 + x[1]**2


if __name__ == '__main__':
    print(numerical_gradient(function_2, np.array([3.0, 4.0])))
    print(numerical_gradient(function_2, np.array([0, 2.0])))
