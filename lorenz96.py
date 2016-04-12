import functools
import numpy as np
import matplotlib.pyplot as plt


def runge_kutta(f, t, x, h):
    k1 = f(t, x)
    k2 = f(t + h / 2, x + k1 / 2)
    k3 = f(t + h / 2, x + k2 / 2)
    k4 = f(t + h, x + k3)

    return x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def lorenz96(t, x, F):
    J = len(x)
    v = np.zeros(J)
    for i in np.arange(J):
        v[i] = -x[(i - 2 + J) % J] * x[(i - 1 + J) % J] + x[(i - 1 + J) % J] * x[(i + 1) % J] - x[i] + F
    return v


def solve_ODE(f, t0, x0, t_end, h):
    step = (int)((t_end - t0) / h)
    q = [[0, []] for i in np.arange(step)]
    q[0] = [t0, x0]
    for i in np.arange(1, step):
        q[i] = [t0 + i * h, runge_kutta(f, t0 + i * h, q[i - 1][1], h)]

    return q


def main():
    J = 40
    # x0 = np.array([np.random.randn() for i in np.arange(J)])
    x0 = np.zeros(J)
    x0[J / 2] = 1e-1

    t_end = 0.4
    h = 1e-3

    q = solve_ODE(functools.partial(lorenz96, F=8.0), 0.0, x0, t_end, h)

    for i, e in enumerate(q):
        if i % 20 == 0:
            plt.plot(e[1])

    plt.show()

main()
