#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpaches
import matplotlib as mpl
import backtrader as bt 
import yfinance as yf


# In[2]:


df = yf.download('BA',start='2021-08-03', end='2022-08-04',progress=False)
df.head


# In[3]:


df = df.loc[:,['Adj Close']]
df.rename(columns={'Adj Close':'adj_close'}, inplace=True)# inplace meaning
df['simple_rtn']=df.adj_close.pct_change()# meaning 
df['log_rtn']=np.log(df.adj_close/df.adj_close.shift(1))


# In[4]:


df.head


# In[5]:


#Following charts present the prices of MS as well as its simple and logarithmic
fig,ax= plt.subplots(3,1 ,figsize=(24,20),sharex=True)
df.adj_close.plot(ax=ax[0])
ax[0].set(title='MS time series',ylabel='Stock price ($)')
df.simple_rtn.plot(ax=ax[1])
ax[1].set(ylabel='Simple return (%)')
df.log_rtn.plot(ax=ax[2])
ax[2].set(xlabel='Date', ylabel='Log return (%)')


# In[6]:


df_rolling = df[['simple_rtn']].rolling(window=22).aggregate(['mean','std'])


df_rolling.columns = df_rolling.columns.droplevel()
df_outliers = df.join(df_rolling)
def indentify_outliers(row, n_sigmas=3) :
    x = row['simple_rtn']
    mu = row['mean']
    sigma = row['std']
    if (x > mu + 3 * sigma) | (x < mu - 3 * sigma) :
        return 1
    else:
        return 0 


# In[7]:


df_outliers['outlier'] = df_outliers.apply(indentify_outliers,axis=1)
outliers = df_outliers.loc[df_outliers['outlier']==1,['simple_rtn']]

fig, ax = plt.subplots(figsize=(18,12))
ax.plot(df_outliers.index,df_outliers.simple_rtn,color='blue',label='Normal')
ax.scatter(outliers.index,outliers.simple_rtn, color='red',label='Anomaly')
ax.set_title("Boeing Company stock returns")
ax.legend(loc='lower right')


# In[8]:


class SmaStrategy (bt.Strategy) :
    params = (('ma_period' , 20),)
    def __init__ (self):
        self.data_close = self.datas[0].close
        self.order = None
        self.price = None
        self.comm = None
        self.sma = bt.ind.SMA(self.datas [0],
                                period=self.params.ma_period)
        


# In[9]:


def log(self, txt):
    dt=self.datas [0].datetime.date(0).isoformat()
    print(f'{dt}, {txt}')
    
def notify_order(self, order):
    if order.status in [order.Submitted, order.Accepted]:
        return
    
    if order.status in [order.Completed]:
        if order.isbuy():
                    self.log(f'BUY EXECUTED --- Price:{order.executed.price:.2f}')# commision #XX
                    
                             
        else:
            self.log(f' SELL EXECUTED --- Price:{order.executed.price:.2f}')
                     

            self.bar_executed = len(self)
                     
    elif order.status in [order.Canceled,order.Margin,order.Rejected]:
                self.log('Order Fatled')
    self.order=None


# In[10]:


def notify_trade(self, trade):
    if not trade.isclosed:
        return
    self.log(f'OPERATION RESULT ---Gross: {trade.pnl:.2f},Net:{trade.pnlcomm:.2f}')
    
def next(self):
    if self.order:
        return
    if not self.position:
        if self.data_close[0] > self.sma[0]:
            self.log(f' BUY CREATED --- Price:{self.data_close[0]:.2f}')
            self.order = self.buy()
    else:
        if self.data_close[0]< self.sma[0]:
                self.log(f' SELL CREATED --- Price:{self.data_close[0]:.2f}')
                self.order = self.sell()


# In[11]:


import yfinance as yf
import backtrader as bt

data = bt.feeds.PandasData(dataname=yf.download('BA', '2021-08-03', '2022-08-04'))

cerebro = bt.Cerebro()
cerebro.adddata(data)


# In[ ]:





# In[12]:


cerebro =bt.Cerebro(stdstats=False)
cerebro.adddata(data)


# In[13]:


cerebro.broker.setcash(10000000.0)
cerebro.addstrategy(SmaStrategy)
cerebro.broker.setcommission(0.001)
cerebro.addsizer(bt.sizers.PercentSizer,percents=10)
cerebro.addobserver(bt.observers.BuySell)
cerebro.addobserver(bt.observers.Value)


# In[14]:


print (f'Starting Portfolio Value: {cerebro.broker.getvalue():.2f}')


# In[15]:


cerebro.run()


# In[16]:


print (f'Starting Portfolio Value')
cerebro. run()
print (f' Final Portfolio Value')


# In[17]:


print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())


# In[18]:


cerebro.run()


# In[19]:


print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())


# #Backtesting
# from __future__ import (
#      absolute_import,
#      division,
#      print_function,
#      unicode_literals,
# )
# 
# import backtrader as bt
# from datetime import datetime
# from fastquant.strategies.base import BaseStrategy
# 
# class MACDStrategy(BaseStrategy):
#     
#     params = (
#     ("fast_period",12),
#     ("slow_period",26),
#     ("signal_period",9),
#     )

# from fastquant.strategies.base import BaseStrategy

# #def __init__ (self) :
#     super().__init__()
# 
# self.fast_period = self.params.fast_period
# self.slow_period = self.params.slow_period
# self.signal_period = self.params.signal_period
# self.commission = self.params.commission
# 
#     if self.strategy_logging:
# print("===Strategy level arguments===")
# print("fast period: ",self.fast_period)
# print ("slow period:",self.slow_period)
# print ("signal period :",self.signal_period)
# 
#     maco_ind = bt.ind.MACD(
#         period_me1=self.fast_period,
#         period_me2=self.slow_period,
#         period_signal=self.signal_period
#     

# 
#     self.macd = macd_ind.macd
#     self.signal =macd_ind.signal
#     self.crossover = bt.ind.CrossOver(
#         self.macd, self.signal
#     )
#     
# def buy_signal(self):
#     return self.crossover> 0
# 
# def sell signal(self) :
#     return self.crossover < 0

# In[20]:


pip install backtesting


# In[21]:


pip install bokeh


# In[22]:


pip install echo


# In[23]:


pip install backtrader


# In[24]:


pip install bt


# In[25]:


pip install ffn 


# In[26]:


import ffn


# In[28]:


import bt
# fetch some data
data = bt.get('BA', start='2021-08-03', end='2022-08-04')
print(data.head())


# In[29]:


# create the strategy
s = bt.Strategy('s1', [bt.algos.RunMonthly(),
                       bt.algos.SelectAll(),
                       bt.algos.WeighEqually(),
                       bt.algos.Rebalance()])


# In[30]:


# create a backtest and run it
test = bt.Backtest(s, data)
res = bt.run(test)


# In[31]:


res.plot()


# In[32]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[33]:


res.plot()


# In[34]:


res.display()


# In[35]:


res.plot_histogram()


# In[36]:


res.plot_security_weights()


# In[37]:


from backtesting import Backtest, Strategy
from backtesting.lib import crossover

from backtesting.test import SMA, BA


class SmaCross(Strategy):
    n1 = 10
    n2 = 20

    def init(self):
        close = self.data.Close
        self.sma1 = self.I(SMA, close, self.n1)
        self.sma2 = self.I(SMA, close, self.n2)

    def next(self):
        if crossover(self.sma1, self.sma2):
            self.buy()
        elif crossover(self.sma2, self.sma1):
            self.sell()


bt = Backtest(BA, SmaCross,
              cash=10000, commission=.002,
              exclusive_orders=True)

output = bt.run()

bt.plot()


# In[38]:


import numpy
import talib

close = numpy.random.random(100)


# In[ ]:


output = talib.SMA(close)


# In[ ]:


from talib import MA_Type

upper, middle, lower = talib.BBANDS(close, matype=MA_Type.T3)


# In[ ]:


output = talib.MOM(close, timeperiod=5)


# In[ ]:


from backtesting import Backtest, Strategy
from backtesting.lib import crossover

from backtesting.test import SMA, BA


class SmaCross(Strategy):
    n1 = 10
    n2 = 20

    def init(self):
        close = self.data.Close
        self.sma1 = self.I(SMA, close, self.n1)
        self.sma2 = self.I(SMA, close, self.n2)

    def next(self):
        if crossover(self.sma1, self.sma2):
            self.buy()
        elif crossover(self.sma2, self.sma1):
            self.sell()


bt = Backtest(BA, SmaCross,
              cash=10000, commission=.002,
              exclusive_orders=True)

output = bt.run()
bt.plot()


# In[ ]:


from backtesting.test import BA

BA.tail()


# In[ ]:


import pandas as pd


def SMA(values, n):
    """
    Return simple moving average of `values`, at
    each step taking into account `n` previous values.
    """
    return pd.Series(values).rolling(n).mean()


# In[ ]:


from backtesting import Strategy
from backtesting.lib import crossover


class SmaCross(Strategy):
    # Define the two MA lags as *class variables*
    # for later optimization
    n1 = 10
    n2 = 20
    
    def init(self):
        # Precompute the two moving averages
        self.sma1 = self.I(SMA, self.data.Close, self.n1)
        self.sma2 = self.I(SMA, self.data.Close, self.n2)
    
    def next(self):
        # If sma1 crosses above sma2, close any existing
        # short trades, and buy the asset
        if crossover(self.sma1, self.sma2):
            self.position.close()
            self.buy()

        # Else, if sma1 crosses below sma2, close any existing
        # long trades, and sell the asset
        elif crossover(self.sma2, self.sma1):
            self.position.close()
            self.sell()


# In[ ]:


get_ipython().run_cell_magic('script', 'echo', '\n    def next(self):\n        if (self.sma1[-2] < self.sma2[-2] and\n                self.sma1[-1] > self.sma2[-1]):\n            self.position.close()\n            self.buy()\n\n        elif (self.sma1[-2] > self.sma2[-2] and    # Ugh!\n              self.sma1[-1] < self.sma2[-1]):\n            self.position.close()\n            self.sell()')


# In[ ]:


from backtesting import Backtest

bt = Backtest(BA, SmaCross, cash=10_000, commission=.002)
stats = bt.run()
stats


# In[ ]:


bt.plot()


# In[ ]:


get_ipython().run_cell_magic('time', '', "\nstats = bt.optimize(n1=range(5, 30, 5),\n                    n2=range(10, 70, 5),\n                    maximize='Equity Final [$]',\n                    constraint=lambda param: param.n1 < param.n2)")


# In[ ]:


stats


# In[ ]:


stats._strategy


# In[ ]:


bt.plot(plot_volume=False, plot_pl=False)


# In[ ]:


stats.tail()


# In[ ]:


stats['_equity_curve'] 


# In[ ]:


stats['_trades']

