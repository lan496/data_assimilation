import numpy as np
import matplotlib.pyplot as plt


class data_assimilation:

    def __init__(self, f, inflation, mask=np.ones(40), delta=1e-5, dt=0.05):
        self.J = 40
        self.dt = dt
        self.f = f
        self.delta = delta
        self.inflation = inflation
        self.mask = mask
        self.N = sum([1 for e in mask if e == 1])

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
            tmp.append(M_j)
        res = np.dstack(tmp)[0]
        return res

    def H(self, x):
        res = np.array(x[self.mask == 1])
        return res

    def H_jacobian(self, x):
        res = np.zeros((self.N, self.J))
        i = 0
        for j, e in enumerate(self.mask):
            if e == 0:
                continue
            res[i, j] = 1.0
            i += 1
        return res

    def kalman_filter(self, xa, Pa, yo, R):
        xf = self.time_evolution(xa)  # J * 1

        M = self.tangent_liner_model(xa)  # J * J
        Pf = np.dot(M, np.dot(Pa, M.T)) * (1.0 + self.inflation)  # J * J

        Hj = self.H_jacobian(xf)  # N * J

        A = np.dot(Hj, np.dot(Pf, Hj.T)) + R  # N * N
        K = np.dot(Pf, np.dot(Hj.T, np.linalg.inv(A)))  # J * N

        xa_next = xf + np.dot(K, self.H(yo - xf))  # J * 1
        Pa_next = np.dot(np.identity(self.J) - np.dot(K, Hj), Pf)  # J * J

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

    def calc_B(self, xa):
        res = np.zeros((self.J, self.J))
        for i in np.arange(len(xa) - 8):
            x48 = np.array(xa[i])
            x24 = np.array(xa[i + 4])
            for j in np.arange(8):
                x48 = self.time_evolution(x48)
            for j in np.arange(4):
                x24 = self.time_evolution(x24)
            dx = x48 - x24
            res += np.dot(np.array([dx]).T, dx.reshape(1, self.J))
        res /= len(xa) - 8
        return res

    def threeD_var(self, xa, yo, B):
        xf = self.time_evolution(xa)  # J * 1

        M = self.tangent_liner_model(xa)  # J * J

        Hj = self.H_jacobian(xf)  # N * J

        A = np.dot(Hj, np.dot(B, Hj.T)) + R  # N * N
        K = np.dot(B, np.dot(Hj.T, np.linalg.inv(A)))  # J * N

        xa_next = xf + np.dot(K, self.H(yo - xf))  # J * 1

        return xa_next

    def xa_with_kalman_filter(self, yt, yo, R):
        xa_0 = np.array(yt[100])
        Pa_0 = 9.0 * np.identity(self.J)

        xa = [xa_0]
        Pa = [Pa_0]

        for i in np.arange(len(yt) - 1):
            if i % 100 == 0:
                print "kalman filter step: " + str(i)
            xa_next, Pa_next = self.kalman_filter(xa[i], Pa[i], yo[i + 1], R)
            xa.append(xa_next)
            Pa.append(Pa_next)

        return xa

    def xa_with_threeD_bar(self, yt, yo, B):
        xa_0 = np.array(yt[100])

        xa = [xa_0]

        for i in np.arange(len(yt) - 1):
            if i % 100 == 0:
                print "threeD var step: " + str(i)
            xa_next = self.threeD_var(xa[i], yo[i + 1], B)
            xa.append(xa_next)

        return xa


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


def find_optimal_inflation(yt, yo, mask):
    candidates = []

    for inflation in np.arange(0.0, 0.1, 0.01):
        print "inflation: " + str(inflation)
        da = data_assimilation(f=lorenz96, inflation=inflation, mask=mask)
        R = np.identity(da.N)
        xa = da.xa_with_kalman_filter(yt, yo, R)
        xa_RMSE = [RMSE(xa[i], yt[i]) for i in np.arange(len(yt))]
        candidates.append((inflation, np.average(xa_RMSE)))

    res = min(candidates, key=(lambda x: x[1]))[0]
    print res
    return res


def find_optimal_B(yt, yo, B0, mask):
    candidates = []

    for a in np.arange(0.1, 2.0, 0.1):
        print "background covariance matrix coefficient: " + str(a)
        da = data_assimilation(f=lorenz96, inflation=inflation, mask=mask)
        xa = da.xa_with_threeD_bar(yt, yo, B0 * a)
        xa_RMSE = [RMSE(xa[i], yt[i]) for i in np.arange(len(yt))]
        candidates.append((a, np.average(xa_RMSE)))

    res = min(candidates, key=(lambda x: x[1]))[0]
    print res
    return res


def main():
    yt = np.loadtxt('truth.txt')
    yo = np.loadtxt('observation.txt')

    inflation = 0.05
    N = 20
    mask = np.ones(40)
    for i in np.arange(N):
        mask[2 * i] = 0
    R = np.identity(N)

    da = data_assimilation(f=lorenz96, inflation=inflation, mask=mask)

    xa_kf = da.xa_with_kalman_filter(yt, yo, R)
    fname = "analysis_" + str(N) + '.txt'
    f_an = open(fname, 'w')
    for e in xa_kf:
        f_an.write(" ".join(map(str, e)) + '\n')
    f_an.close()
    print "create " + fname

    # xa_kf = np.loadtxt(fname)

    B0 = da.calc_B(xa_kf)
    print "calculate background covariance matrix"
    plt.pcolormesh(B0)
    plt.colorbar()

    # xa = da.xa_with_threeD_bar(yt, yo, B0)
    # xa_RMSE = [RMSE(xa[i], yt[i]) for i in np.arange(len(yt))]

    plt.show()


if __name__ == '__main__':
    yt = np.loadtxt('truth.txt')
    yo = np.loadtxt('observation.txt')

    mask = np.ones(40)
    mask[0] = 0
    mask[20] = 0

    # inflation = find_optimal_inflation(yt, yo, mask)
    inflation = 0.07

    da = data_assimilation(f=lorenz96, inflation=inflation, mask=mask)

    R = np.identity(da.N)

    xa_kf = da.xa_with_kalman_filter(yt, yo, R)
    fname = 'analysis.txt'
    f_an = open(fname, 'w')
    for e in xa_kf:
        f_an.write(" ".join(map(str, e)) + '\n')
    f_an.close()
    print "create " + fname

    # xa_kf = np.loadtxt(fname)

    B0 = da.calc_B(xa_kf)
    a = find_optimal_B(yt, yo, B0, mask)
