
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pandas_datareader import data as pdr
from plotly.subplots import make_subplots
import plotly.graph_objects as go

#Get stock data with pandas_datareader

end = dt.datetime.now()
start = dt.datetime(2015,1,1)

df = pdr.get_data_yahoo(['^BVSP', '^GSPC', 'VT'], start, end)
Close = df.Close
Close.head()

#Compute log returns

log_returns = np.log(df.Close/df.Close.shift(1)).dropna()
log_returns

#Calculate daily standard deviation of returns

daily_std = log_returns.std()
daily_std

annualized_vol = daily_std * np.sqrt(252)
annualized_vol 

#Plot histogram of log returns with annualized volatility

fig = make_subplots(rows=3, cols=1)

trace1 = go.Histogram(x=log_returns['^BVSP'], name = 'IBOV')
trace2 = go.Histogram(x=log_returns['^GSPC'], name = 'S&P500')
trace3 = go.Histogram(x=log_returns['VT'], name = 'VT')

fig.add_trace(trace1,1,1)
fig.add_trace(trace2,2,1)
fig.add_trace(trace3,3,1)

fig.update_layout(autosize=False,width=600, height=600, title = "Frequency of log returns",
                  xaxis=dict(title='IBOV Annualized Vol ' + str(np.round(annualized_vol['^BVSP']*100,1))),
                  xaxis2=dict(title='S&P500 Annualized Vol ' + str(np.round(annualized_vol['^GSPC']*100,1))),
                  xaxis3=dict(title='VT Annualized Vol ' + str(np.round(annualized_vol['VT']*100,1)))
                  )

fig.show()

#Trailing volatility over time

Trading_Days = 60
volatility = log_returns.rolling(window=Trading_Days).std()*np.sqrt(Trading_Days)


volatility.plot().update_layout(autosize=False,width=600, height=300)


#Sharpe Ratio
# The Sharpe ratio which was introduced in 1966 by Nobel laurate William F. Sharpe is a measure for calculatind risk-adjusted return.
# It is the average return earned  in excess of the risk free rate per unit off volatility

Rf = 0.10 / 252
sharpe_ratio = (log_returns.rolling(window=Trading_Days).mean() - Rf)*Trading_Days/volatility

plt.title("Sharpe Ratio")
plt.plot(sharpe_ratio)

#Sortino Ratio
# The Sortino ratio is very similar to the Sharpe ratio, the only difference being that where the Sharpe Ratio uses all observatios for calculatin the std dev the Sortino ratio only considers the harmful variance

sortino_vol = log_returns[log_returns<0].rolling(window=Trading_Days, center = True, min_periods =10).std()*np.sqrt(Trading_Days)
sortino_ratio = (log_returns.rolling(window=Trading_Days).mean() - Rf)*Trading_Days/sortino_vol


plt.title("Sortino Vol")
plt.plot(sortino_vol)

plt.title("Sortino Ratio")
plt.plot(sortino_ratio)


#Modigliani Ratio (M2 ratio)

#The Modigliani ratio measures the returns of the portfolio, adjusted for the risk of the portfolio relative to that of some benchmark

m2_ratio = pd.DataFrame()

benchmark_vol = volatility['VT']
for c in log_returns.columns:
    if c!= 'VT':
        m2_ratio[c] = (sharpe_ratio[c]*benchmark_vol/Trading_Days + Rf)*Trading_Days

plt.title("M2 Ratio")
plt.plot(m2_ratio)

#Max Drawdown
# Max drawdown quantifies the steepest decline from the peak to trough observed for an investment.
# This is useful for a number of reasons, mainly the fact that it doesn't rely on the underlying returns being normally distribuited

def max_drawdown(returns):
    cumulative_returns = (1+returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns/peak) - 1
    return drawdown.min()

returns = df.Close.pct_change().dropna()

max_drawdowns = returns.apply(max_drawdown, axis = 0)
max_drawdowns*100

#Calmar Ratio

# Calmar ratio uses max drawdown in the denominator as opposed to standard deviation

calmars = np.exp(log_returns.mean()*252)/abs(max_drawdowns)
calmars.plot.bar()
