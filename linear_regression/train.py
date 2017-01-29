import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def error(x, y, m, b):
    return np.mean((m * x + b - y) ** 2)


def step_gradient(x, y, b, m, lr):
    m_gradient = np.mean(-x * (y - (m * x + b))) * 2
    b_gradient = np.mean(y - (m * x + b)) * 2
    return b - lr * b_gradient, m - lr * m_gradient


def gradient_descent(x, y, b, m, lr, iterations, print_step=10):
    errors = []
    plt.scatter(x, y)
    for iteration in xrange(iterations):
        b, m = step_gradient(x, y, b, m, lr)

        plt.plot(x, m * x + b, color=str(iteration / iterations))
        current_error = error(x, y, m, b)

        errors.append(current_error)
        if not iteration % print_step:
            print "Current Error at step=%s : %s" % (iteration, current_error)

    plt.show()

    plt.plot(range(iterations), errors)
    plt.show()

    return b, m


def main():
    frame = pd.read_csv('data.csv')
    print frame


    LEARNING_RATE = 0.0001

    INIT_B = 0.0
    INIT_M = 0.0

    MAX_ITERATIONS = 1000

    points = frame.as_matrix()
    x = points[:, 0]
    y = points[:, 1]


    b, m = gradient_descent(x, y, INIT_B, INIT_M, LEARNING_RATE, MAX_ITERATIONS)
    print b, m





if __name__ == "__main__":
    main()