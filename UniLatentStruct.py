import numpy as np
import pandas as pd
import os
import math
import numpy.matlib

# file path
file = './Equity'

# use a hashmap to store all the dataframes
data = dict()

# read INVL data in the file path, save it to data
for name in os.listdir(file):
    if 'SIC6' in name or 'SIC2' in name:
        df = pd.read_csv(file+'/'+name)
        df.columns = ['date', name[:4]+'INVL']
        df['date'] = pd.to_datetime(df['date'])
        df[name[:4]+'INVL'] = df[name[:4]+'INVL'].astype(float)
        df = df.dropna()
        data[name[:4]] = df

dta = None
for pt in data.values():
    if dta is None:
        dta = pt
    else:
        dta = pd.merge(dta, pt, how='outer', left_on='date', right_on='date')

dta = dta.interpolate(method='linear', limit_direction='forward', axis=0)
dta.index = dta['date']
dta = dta.iloc[:, 1:]

"""start building this model using SIC2INVL"""
# input data log
LogInputData: bool = True

Y = np.array((dta['SIC6INVL'])).reshape([-1, 1])
K = 3  # hidden states
N = Y.shape[0]  # length of observations
sample = 100
burnin = 100
priorChange = burnin // 2
numStdStart = 2
modelType = 1
priorSlopeVarRescale = 100
slopeVar = np.ones([K, 1]) * priorSlopeVarRescale
slopeMean = np.zeros([K, 1])
sigma2shape = 1
sigma2scale = 1
priorDelta = 0.1
delta = priorDelta
logLikeDraws = np.zeros([burnin+sample+1, 1])
slope = np.zeros([K, 1])    # constant level
sigma2 = np.zeros([K, 1])   # var from above constant level
pY = np.zeros([N, 1])   # predicted observed data, mean value
sigma2Time = np.zeros([N, 1])

P = np.zeros([K, K])    # transition matrix
D = np.zeros([N, 1])    # current realization
nu = np.zeros([K, 1])   # starting probability vector

if LogInputData:
    Y = np.log(Y)

maxY = max(Y.max(), Y.mean() + numStdStart * Y.std())
minY = min(Y.min(), Y.mean() - numStdStart * Y.std())

# filter forward and sample backward algo
fStateGData = np.zeros([N, K])  # f(D_i|F_i)    F_i = {Y_1,...,Y_i}
fStateGDataLag = np.zeros([N, K])   # f(D_i | F_{i - 1})
fStateGDataAll = np.zeros([N, K])   # f(D_i | F_N)

# For Moment Summaries
sfStateGDataAll1 = np.zeros([N, K])
sfStateGDataAll2 = np.zeros([N, K])
sD = np.zeros([N, 2])
spY = np.zeros([N, 2])

sSlope = np.zeros([K, 2])
sP1 = np.zeros([K, K])
sP2 = np.zeros([K, K])
snu = np.zeros([K, 2])
sSigma2 = np.zeros([K, 2])

# Prior for transition matrix - forces longer waiting times
priorP = np.ones([K, K]) * delta + delta * N * np.eye(K)
# # Allow highest state to be short lived
priorP[K-1, K-1] = priorP[K-1, K-1] * delta
### transition matrix not adding up to 1
# priorP = np.ones([K, K]) * (1/K) + np.ones([K, K]) * 2

# INITIAL ESTIMATES
D = np.zeros([N, 1])
pY = np.zeros([N, 1])
mean0fY = Y.mean()
var0fY = Y.var()
for i in range(K):
    slope[i] = mean0fY + (var0fY ** 0.5) * numStdStart * (i * 2 / (K-1) - 1)
sigma2 = var0fY * np.ones([K, 1])
nu = np.ones([K, 1]) / K


### this function cannot function for model 2
def calcLikelihoodAllStates(Y, slope, sigma2, modelType):
    N = Y.shape[0]
    K = slope.shape[0]
    logLike = - 0.5 * np.matlib.repmat(np.log(2*math.pi*sigma2.T), N, 1) - 0.5*np.matlib.repmat(1/sigma2.T, N, 1)*(
        np.matlib.repmat(Y, 1, K) - np.matlib.repmat(slope.T, N, 1)) * (
        np.matlib.repmat(Y, 1, K) - np.matlib.repmat(slope.T, N, 1))
    logOfLikeScalingFactor = logLike.max().max()
    logLike = logOfLikeScalingFactor - logLike
    # leave out a comparison with -200, obtain the greater one
    return np.exp(-logLike).dot(np.exp(-logOfLikeScalingFactor))


fDataGState = calcLikelihoodAllStates(Y, slope, sigma2, 1)
m = np.sort(fDataGState, axis=1)
I = np.argsort(fDataGState, axis=1)
D = I[:, -1].reshape([-1, 1])
for i in range(K):
    pY = pY + (D == i)*slope[i]


def updateTransitionMatrix(D, priorP, P):
    K = P.shape[0]
    nStateTrans = np.zeros([K, K])
    DShift = D[1:]
    DnShift = D[:-1]
    for i in range(K):
        for j in range(K):
            statenShift = (DnShift == i)
            stateShift = (DShift == j)
            nStateTrans[i, j] = (stateShift * statenShift).sum()
    for i in range(K):
        for j in range(K):
            P[i, j] = np.random.gamma(nStateTrans[i, j] + priorP[i, j], 1, 1)
        P[i, :] = P[i, :] / P[i, :].sum()
    return P


P = updateTransitionMatrix(D, priorP, P)


### this function is written by myself
def sliceNormal(miu, sig2, lo, hi):
    trial = miu + pow(sig2, 0.5) * np.random.normal()
    if trial > hi:
        return hi
    if trial < lo:
        return lo
    else:
        return trial


### it is problematic because the lo and hi are not functioning
def genSlope(Y, D, sigma2, slope, slopeMean, slopeVar, minY, maxY):
    N = D.shape[0]
    K = slope.shape[0]
    pY = np.zeros([N, 1])
    for i in range(K):
        state = (D == i)
        varY = 1/(state.sum() / sigma2[i][0] + 1 / pow((slopeVar[i][0]), 0.5))    ### * sigma2[i][0]))
        meanY = ((state * Y).sum() / sigma2[i][0] + slopeMean[i][0] / (slopeVar[i][0] * sigma2[i][0])) * varY
        # slope(i) = meanY(i) + sqrt(varY(i))*randn()
        if i == 0:
            slope[i] = sliceNormal(meanY, varY, minY, slope[i+1])
        elif i == K-1:
            slope[i] = sliceNormal(meanY, varY, slope[i - 1], maxY)
        else:
            slope[i] = sliceNormal(meanY, varY, slope[i - 1], slope[i + 1])
        pY = pY + state * slope[i]
    return pY, slope


pY, slope = genSlope(Y, D, sigma2, slope, slopeMean, slopeVar, minY, maxY)
# print(pY, slope)

sigma2shape = delta * N
sigma2scale = delta * ((Y-pY) * (Y-pY)).sum()


# update sigma2
def genSigma2(Y, D, pY, sigma2, sigma2shape, sigma2scale):
    K = sigma2.shape[0]
    N = Y.shape[0]
    sigma2Time = np.zeros([N, 1])
    SSTime = (Y - pY) * (Y - pY)
    for i in range(K):
        stateIndicator = (D == i)
        N = stateIndicator.sum()
        SS = (SSTime * stateIndicator).sum()
        div = float(np.random.gamma(0.5*N+sigma2shape, (0.5*SS + sigma2scale), 1)[0])
        sigma2[i] = 1.0 / div
    for i in range(D.shape[0]):
        sigma2Time[i] = sigma2[D[i]]
    return sigma2, sigma2Time


sigma2, sigma2Time = genSigma2(Y, D, pY, sigma2, sigma2shape, sigma2scale)


# MCMC analysis
def calcLogLikeHMC(Y, pY, sigma2Time):
    YpY = (Y - pY) * (Y - pY)
    return -0.5 * np.log(sigma2Time).sum() - 0.5 * (1.0/sigma2Time * YpY).sum()


logLikeDraws[0] = calcLogLikeHMC(Y, pY, sigma2Time)


def filterForwardHMMStationary(fDataGState,fStateGDataLag,fStateGData,P,nu):
    N = fDataGState.shape[0]
    for i in range(N):
        if i == 0:
            fStateGDataLag[i, :] = nu.T
        else:
            fStateGDataLag[i, :] = fStateGData[i-1, :].dot(P)
        fStateGData[i,:] = (fDataGState[i, :] * fStateGDataLag[i, :])/(
            fDataGState[i, :].dot(fStateGDataLag[i, :].T))
    return fStateGDataLag, fStateGData


### might go out of index
def loadedDie(prob):
    u = float(np.random.rand(1)[0])
    start = 0
    for i, level in enumerate(prob.cumsum()):
        if start <= u < level:
            return i
        start = level
    # return len(prob) - 1


def calSDA(fDF, P, fDFA, fDFlag, K):
    result = np.zeros([1, K])
    for k in range(K):
        cnt = 0
        for j in range(K):
            cnt += P[k, j] * fDF[k]*fDFA[j]/fDFlag[j]
        result[0, k] = cnt
    return result


def backwardsSampleHMMStationary(fStateGDataAll, D, pY, fStateGData, fStateGDataLag, P, slope):
    N, K = fStateGData.shape
    for ti in range(N):
        i = N - ti
        if i == N:
            fStateGDataAll[N-1, :] = fStateGData[N-1, :]
        else:
            fStateGDataAll[i-1, :] = calSDA(fStateGData[i-1, :], P, fStateGDataAll[i, :], fStateGDataLag[i, :], K)
            # fStateGDataAll[i-1, :] = (np.matlib.repmat(fStateGData[i-1, :].T, K, 1) * P).dot((fStateGDataAll[i, :] / fStateGDataLag[i, :]).T, K)
        D[i-1] = loadedDie(fStateGDataAll[i-1, :].T)
        pY[i-1] = slope[D[i-1]]
    return fStateGDataAll, D, pY

sny = np.zeros([K, 2])

for n in range(burnin+sample):
    if n+1 == priorChange:
        sigma2Scale = delta * sigma2scale
        sigma2Shape = delta * sigma2shape

    # update hidden states: filter forward
    fDataGState = calcLikelihoodAllStates(Y, slope, sigma2, 1)
    fStateGDataLag, fStateGData = filterForwardHMMStationary(fDataGState, fStateGDataLag, fStateGData, P, nu)

    # backward sampling
    fStateGDataAll, D, pY = backwardsSampleHMMStationary(fStateGDataAll, D, pY, fStateGData, fStateGDataLag, P, slope)

    # Update Markov Chain Parameters
    P = updateTransitionMatrix(D, priorP, P)

    # Update Observation or Likelihood Parameters
    # Update slope
    pY, slope = genSlope(Y, D, sigma2, slope, slopeMean, slopeVar, minY, maxY)

    # Update sigma2
    sigma2, sigma2Time = genSigma2(Y, D, pY, sigma2, sigma2shape, sigma2scale)

    # save parameter estimates
    if n+1 > burnin:
        sfStateGDataAll1 = sfStateGDataAll1 + fStateGDataAll
        sfStateGDataAll2 = sfStateGDataAll2 + fStateGDataAll * fStateGDataAll
        sD[:, 0] = sD[:, 0].reshape([1, -1]) + D.reshape([1, -1])
        sD[:, 1] = sD[:, 1].reshape([1, -1]) + (D * D).reshape([1, -1])
        spY[:, 0] = spY[:, 0].reshape([1, -1]) + pY.reshape([1, -1])
        spY[:, 1] = spY[:, 1].reshape([1, -1]) + (pY * pY).reshape([1, -1])

        sSlope[:, 0] = sSlope[:, 0].reshape([1, -1]) + slope.reshape([1, -1])
        sSlope[:, 1] = sSlope[:, 1].reshape([1, -1]) + (slope * slope).reshape([1, -1])
        sP1 = sP1 + P
        sP2 = sP2 + P * P
        snu[:, 0] = snu[:, 0].reshape([1, -1]) + nu.reshape([1, -1])
        sny[:, 1] = snu[:, 1].reshape([1, -1]) + nu.reshape([1, -1])
        sSigma2[:, 0] = sSigma2[:, 0].reshape([1, -1]) + sigma2.reshape([1, -1])
        sSigma2[:, 1] = sSigma2[:, 1].reshape([1, -1]) + (sigma2 * sigma2).reshape([1, -1])

    logLikeDraws[n + 1] = calcLogLikeHMC(Y, pY, sigma2Time)

# generate reports
### snu and sny: not knowing whether sny is a typo
sfStateGDataAll1 = sfStateGDataAll1 / sample
sfStateGDataAll2 = np.power(sfStateGDataAll2/sample - sfStateGDataAll1*sfStateGDataAll1, 0.5)
sD[:, 0] = sD[:, 0].reshape([1, -1])/sample
sD[:, 1] = np.power(sD[:, 1]/sample - sD[:, 0]*sD[:, 0], 0.5)
spY[:, 0] = spY[:, 0]/sample
spY[:, 1] = np.power(spY[:, 1]/sample - spY[:, 0]*spY[:, 0], 0.5)
sSlope[:, 0] = sSlope[:, 0]/sample
sSlope[:, 1] = np.power(sSlope[:, 1]/sample - sSlope[:, 0] * sSlope[:, 0], 0.5)
sP1 = sP1/sample
sP2 = np.power(sP2/sample - sP1 * sP1, 0.5)
snu[:, 0] = snu[:, 0] / sample
sny[:, 1] = np.power(snu[:, 1]/sample - snu[:, 0]*snu[:, 0], 0.5)
sSigma2[:, 0] = sSigma2[:, 0]/sample
sSigma2[:, 1] = np.power(sSigma2[:, 1]/sample - sSigma2[:, 0] * sSigma2[:, 0], 0.5)

import matplotlib.pyplot as plt
result = calcLikelihoodAllStates(Y, slope, sigma2, 1)
new_D = np.argsort(result, axis=1)
new_D = new_D[:, -1].reshape([-1, 1])
plt.plot(df['date'], new_D)
plt.show()
