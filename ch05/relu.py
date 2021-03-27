import numpy as np
from common.layer import Relu


if __name__ == '__main__':
    x = np.array([[1.0, -0.5], [-2.0, 3.0]])
    relu = Relu()
    out = relu.forward(x)
    print(out)

    dout = relu.backward(x)
    print(x)
