from common.differential_function import numerical_diff


def function_1(x):
    return 0.01*x**2 + 0.1*x


if __name__ == '__main__':
    print(numerical_diff(function_1, 5))
    print(numerical_diff(function_1, 10))
