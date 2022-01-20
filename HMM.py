"""
This code use GaussianHMM from hmmlearn.hmm package to implement HMM model.
"""
import numpy as np
import pandas as pd
import os
import math
import numpy.matlib
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM

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


"""IMPLEMENT HMM"""
# construct test sample, here use SIC6 as an example
X_Test = np.array(dta.loc[:, 'SIC6INVL']).reshape(-1, 1)
# take log of the test sample
X_Test = np.log(X_Test)
# generate the model
# n_components means the number of hidden states
model = GaussianHMM(n_components=3, covariance_type='full', n_iter=100)
# fit the model
model.fit(X_Test)

"""PRESENT RESULT"""
print("hidden states", model.n_components)
print("mean")
print(model.means_)
print("covariance matrix")
print(model.covars_)
print("transition matrix")
print(model.transmat_)
# get the corresponding relationship between state and its observable mean
trx = np.argsort(model.means_, axis=0)
# get state number
prdt = model.predict(X_Test)
# get SORTED state number, higher state represents higher observable level
for i in range(len(prdt)):
    prdt[i] = trx[prdt[i]]
print(prdt)
# show the pattern of the result using plot
plt.plot(df['date'], prdt)
plt.show()
