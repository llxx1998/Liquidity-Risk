"""
This code is a step by step implementation of HMM model.
"""
import numpy as np
import pandas as pd
import os
import math
import numpy.matlib
import matplotlib.pyplot as plt

"""READ IN DATA"""
# file path
file = './Equity'
# use a hashmap to store all the dataframes
data = dict()
# read INVL data in the file path, save it to data hashmap
for name in os.listdir(file):
    if 'SIC' == name[:3]:
        df = pd.read_csv(file+'/'+name)
        df.columns = ['date', name[:4]+'INVL']
        df['date'] = pd.to_datetime(df['date'])
        df[name[:4]+'INVL'] = df[name[:4]+'INVL'].astype(float)
        df = df.dropna()
        data[name[:4]] = df
# merge dataframe pointer into a single dataframe
dta = None
for pt in data.values():
    if dta is None:
        dta = pt
    else:
        dta = pd.merge(dta, pt, how='outer', left_on='date', right_on='date')
# use interpolation method to supplement data for certain market
dta = dta.interpolate(method='linear', limit_direction='forward', axis=0)
dta.index = dta['date']
dta = dta.iloc[:, 1:]


"""FUNCTIONS"""
def calcLikelihoodAllStates(Y, slope, sigma2, modelType):
    """
    Calculate the conditional probability of each state 1...K for observables from 1...N
    :param Y: np.array, observables
    :param slope: np.array, slope for state 1...K
    :param sigma2: np.array, variance for state 1...K
    :param modelType: int, indicator for model to use (currently not functioning)
    :return:np.array,  probability conditional on states, N*K matrix
    """
    N = Y.shape[0]
    K = slope.shape[0]
    logLike = - 0.5 * np.matlib.repmat(np.log(2*math.pi*sigma2.T), N, 1) - 0.5*np.matlib.repmat(1/sigma2.T, N, 1)*(
        np.matlib.repmat(Y, 1, K) - np.matlib.repmat(slope.T, N, 1)) * (
        np.matlib.repmat(Y, 1, K) - np.matlib.repmat(slope.T, N, 1))
    logOfLikeScalingFactor = logLike.max().max()
    logLike = logOfLikeScalingFactor - logLike
    return np.exp(-logLike).dot(np.exp(-logOfLikeScalingFactor))


def updateTransitionMatrix(D, priorP, P):
    """
    Use this function to update transition matrix.
    :param D: np.array, hidden states N*1
    :param priorP: np.array, prior transition matrix
    :param P: np.array, transition matrix before making change
    :return: np.array, transition matrix
    """
    K = P.shape[0]
    nStateTrans = np.zeros([K, K])
    DShift = D[1:]
    DnShift = D[:-1]
    # count transition times from i to j
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


def sliceNormal(miu, sig2, lo, hi):
    """
    Sub-function of genSlope. Generate normal random number based on N(miu, sig2).
    The number is constrained in [lo, hi].
    :param miu: float, mean value
    :param sig2: float, variance
    :param lo: float, the lowest value the number can be
    :param hi: flaot, the highest value the number can be
    :return: float, resulting random number from truncated normal distribution.
    """
    trial = miu + pow(sig2, 0.5) * np.random.normal()
    if trial > hi:
        return hi
    if trial < lo:
        return lo
    else:
        return trial


def genSlope(Y, D, sigma2, slope, slopeMean, slopeVar, minY, maxY):
    """
    Use this function to update slope.
    varY and meanY are calculated by formulas on page 34 in the OFR paper.
    :param Y: np.array, observations
    :param D: np.array, hidden states
    :param sigma2: np.array, variance
    :param slope: np.array, mean for slope normal distribution
    :param slopeMean: np.array, mean for slope normal distribution
    :param slopeVar: np.array, variance for slope normal distribution
    :param minY: flaot, minimum constraint for smallest slope
    :param maxY: flaot, minimum constraint for largest slope
    :return: np.array, np.array
    """
    N = D.shape[0]
    K = slope.shape[0]
    pY = np.zeros([N, 1])
    for i in range(K):
        state = (D == i)
        # varY and meanY are calculated by formulas on page 34 in the OFR paper
        varY = 1/(state.sum() / sigma2[i][0] + 1 / pow((slopeVar[i][0]), 0.5))
        meanY = ((state * Y).sum() / sigma2[i][0] + slopeMean[i][0] / (slopeVar[i][0] * sigma2[i][0])) * varY
        # the following is to ensure that lower slope will not have value higher than greater state
        if i == 0:
            slope[i] = sliceNormal(meanY, varY, minY, slope[i+1])
        elif i == K-1:
            slope[i] = sliceNormal(meanY, varY, slope[i - 1], maxY)
        else:
            slope[i] = sliceNormal(meanY, varY, slope[i - 1], slope[i + 1])
        pY = pY + state * slope[i]
    return pY, slope


def genSigma2(Y, D, pY, sigma2, sigma2shape, sigma2scale):
    """
    Use gamma distribution to generate variance for each slope.
    Gamma distribution formulas on page 34 in the OFR paper.
    :param Y: np.array, observations
    :param D: np.array, hidden states
    :param pY: np.array, predicted Y
    :param sigma2: np.array, slope's variance
    :param sigma2shape: np.array, parameter for gamma distribution
    :param sigma2scale: np.array, parameter for gamma distribution
    :return: np.array, np.array
    """
    K = sigma2.shape[0]
    N = Y.shape[0]
    sigma2Time = np.zeros([N, 1])
    SSTime = (Y - pY) * (Y - pY)
    for i in range(K):
        stateIndicator = (D == i)
        N = stateIndicator.sum()
        SS = (SSTime * stateIndicator).sum()
        # Gamma distribution formulas on page 34 in the OFR paper
        div = float(np.random.gamma(0.5*N+sigma2shape, (0.5*SS + sigma2scale), 1)[0])
        sigma2[i] = 1.0 / div
    for i in range(D.shape[0]):
        sigma2Time[i] = sigma2[D[i]]
    return sigma2, sigma2Time


def calcLogLikeHMC(Y, pY, sigma2Time):
    """
    Irrelevant to our study.
    :param Y:
    :param pY:
    :param sigma2Time:
    :return:
    """
    YpY = (Y - pY) * (Y - pY)
    return -0.5 * np.log(sigma2Time).sum() - 0.5 * (1.0/sigma2Time * YpY).sum()


def filterForwardHMMStationary(fDataGState, fStateGDataLag, fStateGData, P, nu):
    """
    This function is an implementation of forward filtering.
    This implementation corresponds to the formula (4) on page 35.
    :param fDataGState: np.array, f(Data t|State t)
    :param fStateGDataLag: np.array, f(State t|Data t-1)
    :param fStateGData: np.array, f(State t|Data t)
    :param P: np.array, transition matrix
    :param nu: np.array, probability vector for states
    :return: np.array, np.array
    """
    N = fDataGState.shape[0]
    for i in range(N):
        if i == 0:
            fStateGDataLag[i, :] = nu.T
        else:
            fStateGDataLag[i, :] = fStateGData[i-1, :].dot(P)
        # formula (4) on page 35
        fStateGData[i, :] = (fDataGState[i, :] * fStateGDataLag[i, :])/(
            fDataGState[i, :].dot(fStateGDataLag[i, :].T))
    return fStateGDataLag, fStateGData


def calSDA(fDF, P, fDFA, fDFlag, K):
    """
    Calculate formula (6) on page 35.
    :param fDF: np.array,
    :param P: np.array, transition matrix, f(D t+1| D t)
    :param fDFA: np.array, f(D t+1|-t+1)
    :param fDFlag: np.array, f(D t+1|-t)
    :param K: int, state number
    :return: np.array
    """
    result = np.zeros([1, K])
    for k in range(K):
        cnt = 0
        for j in range(K):
            # formula (6) on page 35
            cnt += P[k, j] * fDF[k] * fDFA[j]/fDFlag[j]
        result[0, k] = cnt
    return result


def loadedDie(prob):
    """
    Generate state sample based on the cumulative distribution of states.
    :param prob: np.array, probability density for states
    :return:
    """
    u = float(np.random.rand(1)[0])
    start = 0
    for i, level in enumerate(prob.cumsum()):
        if start <= u < level:
            return i
        start = level


def backwardsSampleHMMStationary(fStateGDataAll, D, pY, fStateGData, fStateGDataLag, P, slope):
    """
    The function is an implementation of backward sampling.
    This implementation corresponds to the formula (6) on page 35.
    :param fStateGDataAll: np.array
    :param D: np.array
    :param pY: np.array
    :param fStateGData: np.array
    :param fStateGDataLag: np.array
    :param P: np.array
    :param slope: np.array
    :return: np.array, np.array, np.array
    """
    N, K = fStateGData.shape
    for ti in range(N):
        i = N - ti
        if i == N:
            fStateGDataAll[N-1, :] = fStateGData[N-1, :]
        else:
            # formula (6) on page 35, refer to calSDA function for more illustration
            fStateGDataAll[i-1, :] = calSDA(fStateGData[i-1, :], P, fStateGDataAll[i, :], fStateGDataLag[i, :], K)
        # sampling based on probability distribution
        D[i-1] = loadedDie(fStateGDataAll[i-1, :].T)
        pY[i-1] = slope[D[i-1]]
    return fStateGDataAll, D, pY


"""INITIALIZE PARAMETERS"""
# choose if take the log of the observable variable
LogInputData: bool = True
# use SIC6 as an example
Y = np.array((dta['SIC6INVL'])).reshape([-1, 1])
K = 3  # hidden states
N = Y.shape[0]  # length of observations
# sample + burnin = 200, which is the total number of iteration.
sample = 100
# "Burn-in is a colloquial term that describes the practice of throwing away
# some iterations at the beginning of an MCMC run. This notion says that you
# start somewhere, say at x, then you run the Markov chain for 11 steps
# (the burn-in period) during which you throw away all the data (no output).
# After the burn-in you run normally, using each iterate in your MCMC calculations."
## the above illustration comes from Handbook of Markov Chain Monte Carlo by Steve Brooks
## here, inference second order moments only after burn-in times
burnin = 100
priorChange = burnin // 2
numStdStart = 2
modelType = 1
priorSlopeVarRescale = 100
# the slope follows Gaussian distribution
slopeVar = np.ones([K, 1]) * priorSlopeVarRescale   # mean of slope, 1 ... K state
slopeMean = np.zeros([K, 1])    # variance of slope, 1 ... K state
# the sigma follows gamma distribution
sigma2shape = 1     # gamma distribution shape
sigma2scale = 1     # gamma distribution scale
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
# take log of observables Y
if LogInputData:
    Y = np.log(Y)
# set the maximun level of Y
maxY = max(Y.max(), Y.mean() + numStdStart * Y.std())
# set the minimum level of Y
minY = min(Y.min(), Y.mean() - numStdStart * Y.std())
# filter forward and sample backward algo
## probability mass functions
fStateGData = np.zeros([N, K])  # f(D_i|F_i)    F_i = {Y_1,...,Y_i}
fStateGDataLag = np.zeros([N, K])   # f(D_i | F_{i - 1})
fStateGDataAll = np.zeros([N, K])   # f(D_i | F_N)
# summary statistics of the execution, s means std dev.
## follow the naming rule: s + variable
### currently irrelevant to our study
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
## p.s. it's okay for each line not summing to 1. Because priorP will further be standardized below.
priorP = np.ones([K, K]) * delta + delta * N * np.eye(K)
## Allow highest state to be short lived
priorP[K-1, K-1] = priorP[K-1, K-1] * delta


"""INITIALIZE ESTIMATES"""
# D is the hidden states for Y
D = np.zeros([N, 1])
# pY is the predicted Y based on hidden states
pY = np.zeros([N, 1])
# mean of Y
mean0fY = Y.mean()
# variance of Y
var0fY = Y.var()
# use mean and variance of Y to generate initial estimate for slope
for i in range(K):
    slope[i] = mean0fY + (var0fY ** 0.5) * numStdStart * (i * 2 / (K-1) - 1)
sigma2 = var0fY * np.ones([K, 1])
# probability vector for each states
nu = np.ones([K, 1]) / K
# calculate the conditional probability of each state 1...K for observables from 1...N
fDataGState = calcLikelihoodAllStates(Y, slope, sigma2, 1)
# find the highest probabilities among states for observations from 1...N
m = np.sort(fDataGState, axis=1)
# find the hidden states with highest probability among states for observations from 1...N
I = np.argsort(fDataGState, axis=1)
# decide on hidden states based on I
D = I[:, -1].reshape([-1, 1])
# use hidden state to generate predicted Y
for i in range(K):
    pY = pY + (D == i)*slope[i]
# use hidden state to update transition matrix
P = updateTransitionMatrix(D, priorP, P)
# update slope and pY
pY, slope = genSlope(Y, D, sigma2, slope, slopeMean, slopeVar, minY, maxY)
# initialize sigma2 gamma distribution's parameters
sigma2shape = delta * N
sigma2scale = delta * ((Y-pY) * (Y-pY)).sum()
# update slope's variance
sigma2, sigma2Time = genSigma2(Y, D, pY, sigma2, sigma2shape, sigma2scale)


"""MCMC OPERATIONS"""
logLikeDraws[0] = calcLogLikeHMC(Y, pY, sigma2Time)     # irrelevant to our study
sny = np.zeros([K, 2])  # irrelevant to our study
# iteration for forward filtering and backward sampling
for n in range(burnin+sample):
    # change the prior parameters of the state's variance
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
    # irrelevant to our study: generate parameter estimates of second moment
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
    logLikeDraws[n + 1] = calcLogLikeHMC(Y, pY, sigma2Time)     # irrelevant to our study
# irrelevant to our study
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


"""PRESENT RESULT"""
result = calcLikelihoodAllStates(Y, slope, sigma2, 1)
new_D = np.argsort(result, axis=1)
new_D = new_D[:, -1].reshape([-1, 1])
plt.plot(df['date'], new_D)
plt.show()
