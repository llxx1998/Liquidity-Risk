# read in the sample data base
import pandas as pd
import time
import re

file = './partly_stock.csv'

df = pd.read_csv(file)
#, dtype={'RET': float}, error_bad_lines=False, warn_bad_lines=False)
df['RET'].replace('B', None, inplace=True)
df['RET'].replace('C', None, inplace=True)
print(df.head().T)

# drop duplicates & NaN
df = df.drop_duplicates().dropna()

# date treatment
df['date'] = df['date'].astype(str)
df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
df['PERMNO'] = df['PERMNO'].astype(int).astype(str)
df['RET'] = df['RET'].astype(float)


# transfer sic code into single integer from 0-8.
def SicTrans(x):
    """
    input: SICCD
    output: int 0-9 to distinguish different market. 9 will be excluded later
    """
    return int(x) // 1000


df['SICCD'] = df['SICCD'].apply(lambda x: SicTrans(x)).astype(int)

# exclude industry 9
df = df[df['SICCD'] < 9]
# see SIC's distribution
print(df.groupby(['SICCD']).count().loc[:, ['date']])

# flip the negative prices to positive ones (caused by lagged price)
df['PRC'] = df['PRC'].abs()

# date from past to present
df = df.sort_values(by=['date'], ascending=True)

# specification on parameters
import math
import numpy as np


def Volatility(cal_df):
    stock_ls = cal_df['PERMNO'].unique()
    if not list(stock_ls):
        return 0.01
    n = len(stock_ls)
    num = 0
    for i in stock_ls:
        try:
            std_err = cal_df.loc[cal_df['PERMNO'] == i, 'RET'].values.std()
            t = cal_df['PERMNO'].loc[cal_df['PERMNO'] == i].count()
            num += std_err / pow(t, 0.5)
        except:
            print('volatility exception')
            num += 0.01
        continue
    return num / n


def SicInvMeas(tempDF: pd.DataFrame):
    date_ls = tempDF['date'].unique()
    if not list(date_ls):
        return -1
    n = len(date_ls)
    result = []
    for i in range(20, n):
        try:
            # i-20, i (not included)
            cal_df = tempDF.loc[tempDF['date'] < date_ls[i], :]
            cal_df = cal_df[cal_df['date'] >= date_ls[i - 20]]
            avg_vol = cal_df['VOL'].sum() / 20
            t = date_ls[i - 1]
            price = cal_df.loc[cal_df['date'] == t, 'PRC'].mean()
            sig = Volatility(cal_df)
            measure = CxCal(sig, avg_vol, price)
            result.append([t, measure])
        except:
            print('exception')
            pass
        continue
    return pd.DataFrame(result)


def CxCal(sig, avg_Vol, p):
    XoverV = pow(10, -5.71)  # as introduced in the 57 page of Kyle & Obiz 2014
    Vol_std = 0.02
    const1 = 8.21e-4
    const2 = 2.5e-4
    expt1 = - 1 / 3
    expt2 = 1 / 3
    W_std = 0.02 * 40 * 1e6
    XV_std = 0.01
    return (sig / Vol_std) * (const1 * pow(sig * p * avg_Vol / W_std, expt1) +
                              const2 * pow(sig * p * avg_Vol / W_std, expt2) * XoverV / XV_std)

# calculate different SIC group
## calculate and save as different csv file
for sic in range(9):
    temp = df[df['SICCD'] == sic]
    print(temp)
    SicInvMeas(temp).to_csv('./Equity/SIC' + str(sic) + '.csv', index=False)
