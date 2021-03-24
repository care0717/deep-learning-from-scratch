import os
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.activation_function import softmax

if __name__ == '__main__':
    x = np.array([0.3, 2.9, 4.1])
    y = softmax(x)
    print(y)
