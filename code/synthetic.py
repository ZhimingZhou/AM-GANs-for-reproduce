import os, sys
import numpy as np
from utils import *
import matplotlib.pyplot as plt

if __name__=="__main__":

    matplotlib.rcParams.update({'font.size': 18})

    def assert_sum(p):
        assert abs(sum(p)-1.0)<0.0001

    def one_hot(i, n):
        tmp = np.zeros(n)
        tmp[i] = 1
        return tmp

    def KL(p, q):
        assert_sum(p)
        assert_sum(q)
        tmp = 0
        try:
            for i in range(len(p)):
                if p[i]>0:
                    tmp += p[i]*(np.log(p[i])-np.log(q[i]))
        except:
            assert False
        return tmp

    N = 100
    ymax = 5.0

    P_N = [np.ones(N) / float(N)]

    np.random.seed(12345)
    GaussianA = np.sort(abs(np.random.randn(N)))
    GaussianA /= np.sum(GaussianA)

    GaussianB = np.sort(abs(np.random.randn(N)))
    GaussianB /= np.sum(GaussianB)
    for i in range(int(N//2)):
        tmp = GaussianB[i]
        GaussianB[i] = GaussianB[N-1-i]
        GaussianB[N-1-i] = tmp

    P_N.append(GaussianA)
    P_N.append(GaussianB)

    NN = 1000

    Curves = []
    for test in range(len(P_N)):

        print(np.array2string(np.asarray(P_N[test]), formatter={'float_kind': lambda x: "%.2f" % x}))

        Curve = []
        for m in range(N):

            N_m = np.zeros(N)
            sum_density = 0
            for i in range(m + 1):
                N_m += one_hot(i, N) * P_N[test][i]
                sum_density += P_N[test][i]
            N_m /= sum_density
            assert_sum(N_m)

            KL_mean = 0
            for i in range(m + 1):
                KL_mean += KL(one_hot(i, N), N_m) * P_N[test][i] / sum_density
            print(N, m + 1, KL_mean)

            Curve.append(KL_mean)

        Curves.append(Curve)

        print('\n\n')

    Curve = []
    test = 1
    for m in range(N):

        scores = []

        for k in range(NN):

            sum_density = 0
            N_m = np.zeros(N)
            keep = np.zeros(N)
            for t in range(m + 1):
                i = np.random.randint(0, N - sum(keep))
                count = 0
                for j in range(N):
                    if not keep[j]:
                        if count == i:
                            i = j
                            break
                        count += 1
                sum_density += P_N[test][i]
                N_m += one_hot(i, N) * P_N[test][i]
                keep[i] = 1.0
            assert sum(keep) == m + 1
            N_m /= sum_density
            assert_sum(N_m)

            KL_mean = 0
            for i in range(N):
                if keep[i]:
                    KL_mean += KL(one_hot(i, N), N_m) * P_N[test][i] / sum_density * keep[i]

            scores.append(KL_mean)

        Curve.append(scores)

    #ymax = max(np.max(np.array(Curves)), np.max(np.array(Curve)))

    plt.figure()
    plt.xlim(0, N)
    plt.ylim(0, ymax)
    plt.ylabel('KL_Score')
    plt.errorbar(np.array(range(N)), [mean(scores) for scores in Curve],
                 [[max(scores) - mean(scores) for scores in Curve], [mean(scores) - min(scores) for scores in Curve]],
                 linestyle='--', fmt='kp', markersize=4)
    # plt.title('Random Drop with Gaussian Class Density', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('../drop_random' + str(N) + '.pdf')

    plt.clf()
    plt.plot(Curves[0], 'kp-', markersize=4)
    plt.xlim(0, N)
    plt.ylim(0, ymax)
    plt.ylabel('kl_score')
    #plt.legend(['Uniform'])
    #plt.title('Random Drop with Uniform Class Density', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('../drop_uniform'+str(N)+'.pdf')

    plt.clf()
    plt.plot(Curves[1], 'kp-', markersize=4)
    plt.plot(Curves[2], 'bp-', markersize=4)
    plt.xlim(0, N)
    plt.ylim(0, ymax)
    plt.ylabel('kl_score')
    plt.legend(['Gaussian-Inc', 'Gaussian-Dec'])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('../drop_gaussian'+str(N)+'.pdf')

