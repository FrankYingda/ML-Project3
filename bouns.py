import numpy as np
from numpy.matlib import repmat
import matplotlib
#matplotlib.use('PDF')
import matplotlib.pyplot as plt
import sys
from scipy.io import loadmat
import time

OFFSET = 2

def l2distance(X, Z=None):
    if Z is None:
        n, d = X.shape
        s1 = np.sum(np.power(X, 2), axis=1).reshape(-1,1)
        D1 = -2 * np.dot(X, X.T) + repmat(s1, 1, n)
        D = D1 + repmat(s1.T, n, 1)
        np.fill_diagonal(D, 0)
        D = np.sqrt(np.maximum(D, 0))
    else:
        n, d = X.shape
        m, _ = Z.shape
        s1 = np.sum(np.power(X, 2), axis=1).reshape(-1,1)
        s2 = np.sum(np.power(Z, 2), axis=1).reshape(1,-1)
        D1 = -2 * np.dot(X, Z.T) + repmat(s1, 1, m)
        D = D1 + repmat(s2, n, 1)
        D = np.sqrt(np.maximum(D, 0))
    return D


def toydata(OFFSET, N):

    NHALF = int(np.ceil(N / 2))
    x = np.random.randn(N, 2)
    #print(x, "\n")
    x[NHALF:, :] += OFFSET

    y = np.ones(N)
    y[NHALF:] *= 2

    jj = np.random.permutation(N)
    return x[jj, :], y[jj]
    #return x, y


def computeybar(xTe, OFFSET):

    n, temp = xTe.shape
    ybar = np.zeros(n)

    # Feel free to use the following function to compute p(x|y), or not
    # normal distribution is default mu = 0, sigma = 1.
    normpdf = lambda x, mu, sigma: np.exp(-0.5 * np.power((x - mu) / sigma, 2)) / (np.sqrt(2 * np.pi) * sigma)

    pxy1 = np.product(normpdf(xTe, 0, 1), axis =1)
    # print("p(x|y1)", pxy1, "\n")
    pxy2 = np.product(normpdf(xTe, OFFSET, 1), axis=1)
    # print("p(x|y2)", pxy2, "\n")

    py1 = 0.5
    py2 = 0.5

    px = pxy1*py1 + pxy2*py2
    # print("px", px, "\n")

    py1x = pxy1 * 0.5 / px
    # print("p(y1|x)", py1x, "\n")
    py2x = pxy2 * 0.5 / px
    # print("p(y2|x)", py2x, "\n")
    #
    # for i in range(0,n):
    #     ybar[i] = max((pxy1[i] * 0.5 / px[i]), (pxy2[i] * 0.5 / px[i]))
    #
    # # py1x = pxy1 * 0.5 / px
    # # pyx2 = pxy2 * 0.5 / px

    ybar = py1x + py2x * 2
    # print(ybar)

    return ybar

# xTe, yTe = toydata(OFFSET, 10)
# print(xTe)
# print(yTe)
#
# yybar = computeybar(xTe,OFFSET)
#
# #print(yybar)


# compute Bayes Error -- Noise
# ybar = computeybar(xTe, OFFSET)
# predictions = np.round(ybar)
# errors = predictions != yTe
# err = errors.sum() / len(yTe) * 100
# print('Error of Bayes classifier: %.1f%%.' % err)
# #
# # plot data
# i1 = yTe == 1
# i2 = yTe == 2
# plt.figure(figsize=(10,6))
# plt.scatter(xTe[i1, 0], xTe[i1, 1], c='r', marker='o')
# plt.scatter(xTe[i2, 0], xTe[i2, 1], c='b', marker='o')
# plt.scatter(xTe[errors, 0], xTe[errors, 1], c='k', s=100, alpha=0.2)
# plt.title("Plot of data (misclassified points highlighted)")
# plt.show()


def kregression(xTr, yTr, sigma=0.1, lmbda=0.01):
    kernel = lambda x, z: np.power(1 + (np.power(l2distance(x, z), 2) / (2 * np.power(sigma, 2))), -4)
    ridge = lambda K, lmbda2: K + lmbda * np.eye(K.shape[0], K.shape[1])
    beta = np.linalg.solve(ridge(kernel(xTr, xTr), lmbda), yTr)

    fun = lambda Xt: np.dot(kernel(Xt, xTr), beta)
    return fun

# prediction = fun(xTe)
# print(prediction)


def computehbar(xTe, sigma, lmbda, Nsmall, NMODELS, OFFSET):
    n = xTe.shape[0]
    hbar = np.zeros(n)

    for j in range(NMODELS):
        [xTr, yTr] = toydata(OFFSET, Nsmall)
        fun = kregression(xTr, yTr, sigma, lmbda)
        hbar = hbar + fun(xTe)

    hbar /= NMODELS
    return hbar


def computevariance(xTe, sigma, lmbda, hbar, Nsmall, NMODELS, OFFSET):
    n = xTe.shape[0]
    variance = np.zeros(n)

    for j in range(NMODELS):
        [xTr, yTr] = toydata(OFFSET, Nsmall)
        fun = kregression(xTr, yTr, sigma, lmbda)
        hd =fun(xTe)
        variance += (hd - hbar) ** 2

    variance = np.mean(variance) / NMODELS
    return variance

#
# xTe, yTe = toydata(OFFSET, 10)
# sigma = 4
# lmbdas = 0.5
# lmbda = 2 ** lmbdas
# Nsmall = 10
# NMODELS = 100
# hbar = computehbar(xTe, sigma, lmbda, Nsmall, NMODELS, OFFSET)
# print(hbar, "\n")
# variance = computevariance(xTe, 4, 0.1, hbar, Nsmall, NMODELS, OFFSET)
# print(variance, "\n")


#
#
# # how big is the training set size N
Nsmall = 10
# how big is a really big data set (approx. infinity)
Nbig = 10000
# how many models do you want to average over
NMODELS = 100
# What regularization constants to evaluate
lmbdas = np.arange(-6, 0 + 0.5, 0.5)
# what is the kernel width?
sigma = 4

# we store
Nlambdas = len(lmbdas)
lbias = np.zeros(Nlambdas)
lvariance = np.zeros(Nlambdas)
ltotal = np.zeros(Nlambdas)
lnoise = np.zeros(Nlambdas)
lsum = np.zeros(Nlambdas)


# Different regularization constant classifiers
for md in range(Nlambdas):
    lmbda = 2 ** lmbdas[md]
    # use this data set as an approximation of the true test set
    xTe, yTe = toydata(OFFSET, Nbig)

    # Estimate AVERAGE ERROR (TOTAL)
    total = 0
    for j in range(NMODELS):
        xTr2, yTr2 = toydata(OFFSET, Nsmall)
        fsmall = kregression(xTr2, yTr2, sigma, lmbda)
        total += np.mean((fsmall(xTe) - yTe) ** 2)
    total /= NMODELS

    # Estimate Noise
    ybar = computeybar(xTe, OFFSET)
    noise = np.mean((yTe - ybar) ** 2)

    # Estimate Bias
    hbar = computehbar(xTe, sigma, lmbda, Nsmall, NMODELS, OFFSET)
    bias = np.mean((hbar - ybar) ** 2)
    #
    # Estimating VARIANCE
    variance = computevariance(xTe, sigma, lmbda, hbar, Nsmall, NMODELS, OFFSET)

    # print and store results
    lbias[md] = bias
    lvariance[md] = variance
    print(lvariance[md])
    ltotal[md] = total
    lnoise[md] = noise
    lsum[md] = lbias[md] + lvariance[md] + lnoise[md]
    print(
        'Regularization λ=2^%2.1f: Bias: %2.4f Variance: %2.4f Noise: %2.4f Bias+Variance+Noise: %2.4f Test error: %2.4f'
        % (lmbdas[md], lbias[md], lvariance[md], lnoise[md], lsum[md], ltotal[md]))

# plot results
plt.figure(figsize=(10,6))
plt.plot(lbias[:Nlambdas],c='r',linestyle='-',linewidth=2)
plt.plot(lvariance[:Nlambdas],c='k', linestyle='-',linewidth=2)
plt.plot(lnoise[:Nlambdas],c='g', linestyle='-',linewidth=2)
plt.plot(ltotal[:Nlambdas],c='b', linestyle='-',linewidth=2)
plt.plot(lsum[:Nlambdas],c='k', linestyle='--',linewidth=2)

plt.legend(["Bias","Variance","Noise","Test error","Bias+Var+Noise"]);
plt.xlabel("Regularization $\lambda=2^x$",fontsize=18);
plt.ylabel("Squared Error",fontsize=18);
plt.xticks([i for i in range(Nlambdas)],lmbdas);
plt.show()

