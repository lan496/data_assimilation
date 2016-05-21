import numpy as np
import matplotlib.pyplot as plt


class data_assimilation:

    def __init__(self, f, inflation, J=40, dt=0.05, delta=1e-5):
        self.J = J
        self.dt = dt
        self.f = f
        self.delta = delta
        self.inflation = inflation

    def time_evolution(self, x):
        res = runge_kutta(self.f, x, self.dt)
        return res

    def tangent_liner_model(self, x):
        delta = self.delta
        tmp = []
        for j in np.arange(self.J):
            e_j = np.zeros(self.J)
            e_j[j] = 1.0
            M_j = (self.time_evolution(x + delta * e_j) - self.time_evolution(x)) / delta
            # M_j = (self.time_evolution(x + 0.5 * delta * e_j) - self.time_evolution(x - 0.5 * delta * e_j)) / delta
            tmp.append(M_j)
        res = np.dstack(tmp)[0]
        return res

    def kalman_filter(self, xa, Pa, yo, R):
        xf = self.time_evolution(xa)

        M = self.tangent_liner_model(xa)
        Pf = np.dot(M, np.dot(Pa, M.T)) * (1.0 + self.inflation)

        # K = np.dot(Pf, np.linalg.inv(Pf + R))
        K = np.linalg.solve((Pf + R).T, Pf.T).T

        xa_next = xf + np.dot(K, yo - xf)
        Pa_next = np.dot(np.identity(self.J) - K, Pf)

        return xa_next, Pa_next

    def create_data(self):
        x0 = np.random.rand(40) * 16.0

        spinup_step = int(365 / 5 / self.dt)
        for i in np.arange(spinup_step):
            x0 = self.time_evolution(x0)

        f_tr = open('truth.txt', 'w')
        f_ob = open('observation.txt', 'w')

        observe_step = spinup_step
        for i in np.arange(observe_step):
            f_tr.write(" ".join(map(str, x0)) + '\n')
            f_ob.write(" ".join(map(str, x0 + np.random.randn(self.J))) + '\n')
            x0 = self.time_evolution(x0)

        f_tr.close()
        f_ob.close()

    def B(self, yo, R, sample_size):
        res = np.zeros((self.J, self.J))
        for i in np.arange(sample_size):
            if i % 100 == 0:
                print i
            idx = np.random.randint(len(yo) - 10)
            x48 = np.array(yo[idx - 200])
            x24 = np.array(yo[idx - 200])
            Pa_48 = 9.0 * np.identity(self.J)
            Pa_24 = 9.0 * np.identity(self.J)

            for j in np.arange(8):
                x48, Pa_48 = self.kalman_filter(x48, Pa_48, yo[idx + 1 + j], R)

            for j in np.arange(4):
                x24, Pa_24 = self.kalman_filter(x24, Pa_24, yo[idx + 5 + j], R)

            dx = x48 - x24
            res += np.dot(np.array([dx]).T, dx.reshape(1, self.J))
        res /= sample_size
        return res

    def threeD_var(self, xa, yo, K):
        xf = self.time_evolution(xa)

        M = self.tangent_liner_model(xa)

        # K = np.dot(B, np.linalg.inv(B + R))

        xa_next = xf + np.dot(K, yo - xf)

        return xa_next


def lorenz96(x):
    J = len(x)
    res = np.array([x[i - 1] * (x[(i + 1) % J] - x[i - 2]) - x[i] + 8.0 for i in np.arange(J)])
    return res


def runge_kutta(f, x, h):
    k1 = f(x)
    k2 = f(x + 0.5 * h * k1)
    k3 = f(x + 0.5 * h * k2)
    k4 = f(x + h * k3)
    res = x + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return res


def RMSE(xo, xt):
    J = len(xo)
    return np.linalg.norm(xo - xt) / np.sqrt(J)


def estimate(J, yt, yo, inflation, delta=1e-5):
    da = data_assimilation(lorenz96, J=J, delta=delta, inflation=inflation)

    R = np.identity(J)
    xa_0 = np.array(yt[100])
    Pa_0 = 9.0 * np.identity(J)

    xa = [xa_0]
    Pa = [Pa_0]

    L = len(yt)
    for i in np.arange(L - 1):
        xa_next, Pa_next = da.kalman_filter(xa[i], Pa[i], yo[i + 1], R)
        xa.append(xa_next)
        Pa.append(Pa_next)
        if i % 100 == 0:
            print i

    xa_RMSE = [RMSE(xa[i], yt[i]) for i in np.arange(L)]
    """
    plt.xlabel('time')
    plt.ylabel('x[0]')
    plt.title('x[0] (no multicative inflation)')

    plt.plot([xa[i][0] for i in np.arange(500)], label='xa')
    plt.plot([yt[i][0] for i in np.arange(500)], label='truth')
    plt.legend()
    plt.show()
    """
    return xa_RMSE


def threeD_var_test(J, yt, yo, inflation, sample_size):
    da = data_assimilation(lorenz96, J=J, inflation=inflation)

    R = np.identity(J)
    xa_0 = np.array(yt[100])

    xa = [xa_0]

    B = da.B(yo, R, sample_size)
    K = np.linalg.solve((B + R).T, B.T).T

    L = len(yt)
    for i in np.arange(L - 1):
        xa_next = da.threeD_var(xa[i], yo[i + 1], K)
        xa.append(xa_next)
        if i % 100 == 0:
            print i
    xa_RMSE = [RMSE(xa[i], yt[i]) for i in np.arange(L)]
    return xa_RMSE


if __name__ == '__main__':
    f_tr = open('truth.txt', 'r')
    f_ob = open('observation.txt', 'r')

    yt = []
    yo = []

    for line in f_tr:
        yt.append(np.array(map(float, line.split())))
    for line in f_ob:
        yo.append(np.array(map(float, line.split())))

    xa_RMSE = estimate(40, yt, yo, 0.02)

    plt.xlabel('time')
    plt.ylabel('RMSE')
    plt.yscale('log')

    plt.plot(xa_RMSE)
    plt.show()

    """
    inflation = 0.05
    for sample_size in [100, 400, 1400]:
        xa_RMSE = threeD_var_test(40, yt, yo, inflation, sample_size)
        plt.plot(xa_RMSE, label='sample size = ' + str(sample_size))

    plt.legend()
    plt.show()
    # threeD_var_test(40, yt, yo, 0.05)
    """
