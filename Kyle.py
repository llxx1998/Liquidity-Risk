import pandas as pd


"""CLEAN DATA"""
# read in data
file = './partly_stock.csv'
df = pd.read_csv(file)
# replace non-numeric observation of return rate
df['RET'].replace('B', None, inplace=True)
df['RET'].replace('C', None, inplace=True)
print(df.head().T)
# drop duplicates & NaN
df = df.drop_duplicates().dropna()
# date treatment: turn date into pd.datetime format
df['date'] = df['date'].astype(str)
df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
# turn PERMNO (identifier of stock) into string type
df['PERMNO'] = df['PERMNO'].astype(int).astype(str)
# make sure return rates are float
df['RET'] = df['RET'].astype(float)


"""CLASSIFY BY SIC CODE"""
# transfer sic code into single integer from 0-8.
def SicTrans(x):
    """
    input: SICCD
    output: int 0-9 to distinguish different market. 9 will be excluded later
    """
    return int(x) // 1000


df['SICCD'] = df['SICCD'].apply(lambda x: SicTrans(x)).astype(int)
# exclude industry 9 (as implemented in the paper)
df = df[df['SICCD'] < 9]
# see SIC's distribution
print(df.groupby(['SICCD']).count().loc[:, ['date']])
# flip the negative prices to positive ones (according to the setting of CRSP data)
df['PRC'] = df['PRC'].abs()
# sort date from past to present
df = df.sort_values(by=['date'], ascending=True)


"""CALCULATE PRICE INVARIANT MEASURE"""


def Volatility(cal_df):
    """
    Use this function to generate volatility data from daily return.
    Utilize daily return rate variable.
    This function is different from bond which has an additional module to calculate daily return rate.
    :param cal_df: pd.DataFrame, transfer in the data for corresponding period
    :return: float, volatility
    """
    # generate list of stocks
    stock_ls = cal_df['PERMNO'].unique()
    # if the list is empty, return 0.01 (standard volatility)
    if not list(stock_ls):
        return 0.01
    n = len(stock_ls)
    # initialize num as 0 for adding up volatility from different stock.
    # Use this to calculate average volatility
    num = 0
    for i in stock_ls:
        try:
            # calculate the volatility for stock i in stock_ls
            std_err = cal_df.loc[cal_df['PERMNO'] == i, 'RET'].values.std()
            # standardize the volatility by dividing t^0.5 to make it daily volatility
            t = cal_df['PERMNO'].loc[cal_df['PERMNO'] == i].count()
            num += std_err / pow(t, 0.5)
        except:
            print('volatility exception')
            num += 0.01
        continue
    return num / n


def SicInvMeas(tempDF: pd.DataFrame):
    """
    This function takes the cleaned data set filtered by SIC to generate price invariant measure.
    :param tempDF: pd.DataFrame, data for a certain SIC code
    :return: pd.DataFrame, liquidity measure result
    """
    # generate date series to select from, because each date corresponds to multiple observations
    date_ls = tempDF['date'].unique()
    # treatment for empty input dataframe
    if not list(date_ls):
        return -1
    # length of date number
    n = len(date_ls)
    result = []
    for i in range(20, n):
        try:
            """i-20, i (not included)
            according to the paper:
            price averages the price on a certain day
            volatility include 1 month for calculation, which is 20 days
            volume takes the average amount during 1 month (20 days)"""
            # cal_df select the corresponding 1-month period dataframe
            cal_df = tempDF.loc[tempDF['date'] < date_ls[i], :]
            cal_df = cal_df[cal_df['date'] >= date_ls[i - 20]]
            # calculate average daily volume
            avg_vol = cal_df['VOL'].sum() / 20
            t = date_ls[i - 1]
            # calculate price on the corresponding day
            price = cal_df.loc[cal_df['date'] == t, 'PRC'].mean()
            # calculate daily return volatility and take average
            # refer to volatility function for more detail
            sig = Volatility(cal_df)
            # use the above variables to generate price invariant measure (INVL)
            measure = CxCal(sig, avg_vol, price)
            result.append([t, measure])
        except:
            print('exception')
            pass
        continue
    return pd.DataFrame(result)


def CxCal(sig, avg_Vol, p):
    """
    This function uses volatility, average trading volume, price
    to generate the price invariant measure of liquidity
    :param sig: float, volatility (sigma)
    :param avg_Vol: float, volume
    :param p: float, average price
    :return: float, price invariant measure
    """
    # the following variables are parameters for calculating liquidity measure.
    XoverV = pow(10, -5.71)
    Vol_std = 0.02
    const1 = 8.21e-4
    const2 = 2.5e-4
    expt1 = - 1 / 3
    expt2 = 1 / 3
    W_std = 0.02 * 40 * 1e6     # W* in the paper, a scaler for W
    XV_std = 0.01
    return (sig / Vol_std) * (const1 * pow(sig * p * avg_Vol / W_std, expt1) +
                              const2 * pow(sig * p * avg_Vol / W_std, expt2) * XoverV / XV_std)


# calculate different SIC group's price invariant liquidity measure
## calculate and save as different csv file
for sic in range(9):
    # select data for different SIC code
    temp = df[df['SICCD'] == sic]
    print(temp)
    # calculate the liquidity measure
    SicInvMeas(temp).to_csv('./Equity/SIC' + str(sic) + '.csv', index=False)
