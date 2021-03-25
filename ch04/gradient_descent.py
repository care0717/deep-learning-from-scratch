from common.differential_function import gradient_descent
import numpy as np


def function_2(x):
    return x[0]**2 + x[1]**2


if __name__ == '__main__':
    print(gradient_descent(function_2, np.array(
        [3.0, 4.0]), lr=0.1, step_num=100))
    print(gradient_descent(function_2, np.array([0, 2.0]), lr=0.05))
