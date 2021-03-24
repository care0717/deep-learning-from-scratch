import os
import sys
import matplotlib.pylab as plt
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.activation_function import sigmoid

if __name__ == '__main__':
    x = np.arange(-5.0, 5.0, 0.1)
    y = sigmoid(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1) # y 軸の範囲を指定
    plt.show()

