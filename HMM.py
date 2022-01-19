import random

import numpy as np
import pandas as pd
import os
import math
import numpy.matlib

# file path
file = './Equity'

# use a hashmap to store all the dataframes
data = dict()
np.random.seed(1234)

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

# from hmmlearn import hmm
from hmmlearn.hmm import GaussianHMM

X_Test = np.array(dta.loc[:, 'SIC6INVL']).reshape(-1, 1)
X_Test = np.log(X_Test)
print(X_Test.shape)
model = GaussianHMM(n_components=3, covariance_type='full', n_iter=100)
model.fit(X_Test)
print("hidden states", model.n_components)  #
print("mean")
print(model.means_)
print("covariance matrix")
print(model.covars_)
print("transition matrix")
print(model.transmat_)
print(model.predict(X_Test))
trx = np.argsort(model.means_, axis=0)
prdt = model.predict(X_Test)
for i in range(len(prdt)):
    prdt[i] = trx[prdt[i]]
print(prdt)

import matplotlib.pyplot as plt
plt.plot(df['date'], prdt)
plt.show()
