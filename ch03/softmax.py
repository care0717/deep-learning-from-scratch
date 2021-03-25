from common.activation_function import softmax
import numpy as np

if __name__ == '__main__':
    x = np.array([0.3, 2.9, 4.1])
    y = softmax(x)
    print(y)
