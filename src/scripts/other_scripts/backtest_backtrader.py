from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import datetime as dt
from dateutil.relativedelta import relativedelta

import os.path
import sys
import backtrader as bt
from scripts.other_scripts.kalman_filter_rets import CumulativeReturnsKF
import numpy as np
import pyfolio as pf


# Variables

# Strategy Parameters
TRADE_QTY = 100
ROLLING_WINDOW = 60
BAR_SIZE = '1 hour'
DURATION_STR = '1 D'

# Kalman Parameters
Q_VAR = 1e-5  # Process Noise
R_VAR = 1e-4  # Measurement Noise
ENTRY_THRESHOLD = 2.0  # Z-score to enter trade
EXIT_THRESHOLD = 0.0   # Z-score to exit



# Specify the ticker we wish to backtest
ticker1 = 'XOM'
ticker2 = 'CVX'

class KalmanIndicator(bt.Indicator):
    lines = ('z_score', 'beta')
    plotinfo = dict(subplot=True)
    
    def __init__(self):
        self.kf = CumulativeReturnsKF(Q=Q_VAR, R=R_VAR, window=ROLLING_WINDOW)

    def next(self):
        # data0 is X (Independent), data1 is Y (Dependent)
        x = np.log(self.data0[0])
        y = np.log(self.data1[0])
        
        # Update KF and store in lines for plotting
        _, z_score, _, _, _ = self.kf.update(x, y)
        self.lines.z_score[0] = z_score

class PairStrategyBT(bt.Strategy):

    def log(self, txt, dt=None):
        '''Logging function for this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        '''Keep a reference to the "close" line in the data[0] dataseries'''
        self.stock_x = self.datas[0].close
        self.stock_y = self.datas[1].close
        
        # Initialize the Indicator (Calculates before next() runs)
        self.kalman = KalmanIndicator(self.datas[0], self.datas[1])

        # Tracks pending orders, buy price and commission
        self.order = None
        self.buyprice = None
        self.buycomm = None

    def start(self):
        # Always use datas[0] as master to avoid KeyError (forward reference)
        # Disable plotylimited to allow the axis to auto-scale to fit both assets
        self.datas[0].plotinfo.plotylimited = False
        
        self.datas[1].plotinfo.plotmaster = self.datas[0]
        self.datas[1].plotinfo.sameaxis = True

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f"BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f" %
                        (order.executed.price,
                        order.executed.value,
                        order.executed.comm))
            elif order.issell():
                self.log(f"SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f" %
                        (order.executed.price,
                        order.executed.value,
                        order.executed.comm))
            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        # Reset orders
        self.order = None
        
    def next(self):
        '''Define strategy logic here, we will be ignoring the first 3 bars for the kalman filter to intialize'''
        # Simply log the closing price of the series from the reference
        self.log(f'{ticker2} Close, %.2f' % self.stock_y[0])
        self.log(f'{ticker1} Close, %.2f' % self.stock_x[0])
        
        # Get current Z-Score from Indicator
        z_score = self.kalman.z_score[0]
        current_beta = self.kalman.beta[0]

        # Check if order is pending... if yes
        if self.order:
            return

        # Check if we are in the market
        if not self.position:
            
            # If not currently in market, enter if...
            # Note: KalmanIndicator handles warm-up internally (returns 0.0 initially)
            if z_score >= ENTRY_THRESHOLD:
                self.log('SELL CREATE, %.2f' % self.stock_y[0])
                self.order = self.sell(data=self.datas[1]) # Need to use data feed object self.datas[1] not self.stock_y[0]

                self.log('BUY CREATE, %.2f' % self.stock_x[0])
                self.order = self.buy(data=self.datas[0])  

            elif z_score <= -ENTRY_THRESHOLD:
                self.log('SELL CREATE, %.2f' % self.stock_x[0])
                self.order = self.sell(data=self.datas[0])

                self.log('BUY CREATE, %.2f' % self.stock_y[0])
                self.order = self.buy(data=self.datas[1])
        else:
            # If currently in market, exit if...
            if z_score <= EXIT_THRESHOLD:
                self.log('BUY CREATE, %.2f' % self.stock_y[0])
                self.order = self.buy(data=self.datas[1])

                self.log('SELL CREATE, %.2f' % self.stock_x[0])
                self.order = self.sell(data=self.datas[0])

            if z_score >= -EXIT_THRESHOLD:
                self.log('SELL CREATE, %.2f' % self.stock_y[0])
                self.order = self.sell(data=self.datas[1])

                self.log('BUY CREATE, %.2f' % self.stock_x[0])
                self.order = self.buy(data=self.datas[0])

        


if __name__ == '__main__':
    # Create a cerebro entity
    # stdstats=False removes the default observers (Broker, Trades, BuySell) to declutter
    cerebro = bt.Cerebro(stdstats=False)

    # Determines which the folder where this script (backtest.py) lives
    modpath = os.path.dirname(os.path.abspath(sys.argv[0]))

    # Point to the specific file relative to the script
    # This goes Up 2 levels (../../) -> into 'datas' -> finds '{ticker}_Au.csv'
    datapath = os.path.join(modpath, f'..\\..\\datas\\{ticker1}_Au.csv') # path to data for ticker 1
    datapath1 = os.path.join(modpath, f'..\\..\\datas\\{ticker2}_Au.csv') # path to data for ticker 2

    # # Tells us if the csv file of data we are searching for exists
    # print(f"Looking for file at: {datapath}")
    # print(f"File exists? {os.path.exists(datapath)}")

    # Gets current date and goes back 1 month 
    end_time = dt.datetime.now() # Current Date
    start_time = dt.datetime.now() - relativedelta(years=5) # Back 5 years 

    time_tuple1 = start_time.timetuple() # Creates a tuple for the time format in an extended format
    start_tuple = (time_tuple1[0], time_tuple1[1], time_tuple1[2]) # Specifies the year, month, day only 

    time_tuple2 = end_time.timetuple()
    end_tuple = (time_tuple2[0], time_tuple2[1], time_tuple2[2])
    
    # Get data feed
    data = bt.feeds.GenericCSVData(
        dataname=datapath,
        
        fromdate=dt.datetime(time_tuple1[0], time_tuple1[1], time_tuple1[2]),
        todate=dt.datetime(time_tuple2[0], time_tuple2[1], time_tuple2[2]),

        dtformat=('%Y%m%d'),

        datetime=0,
        open=1,
        high=2,
        low=3,
        close=4,
        volume=5,
        openinterest=-1
    )

    # Disable volume plotting for clearer graph
    data.plotinfo.plotvolume = False

    data1 = bt.feeds.GenericCSVData(
        dataname=datapath1,

        fromdate=dt.datetime(time_tuple1[0], time_tuple1[1], time_tuple1[2]),
        todate=dt.datetime(time_tuple2[0], time_tuple2[1], time_tuple2[2]),

        dtformat=('%Y%m%d'),

        datetime=0,
        open=1,
        high=2,
        low=3,
        close=4,
        volume=5,
        openinterest=-1
    )

    # Disable volume plotting for clearer graph
    data1.plotinfo.plotvolume = False
    
    # Disable plotting of the original data1 so we only see the scaled version overlay
    # data1.plotinfo.plot = False

    # Add the Data Feed to Cerebro
    cerebro.adddata(data) # becomes self.datas[0]
    cerebro.adddata(data1) # becomes self.datas[1]

    # Set our desired cash start
    cerebro.broker.setcash(100000.0)

    # Set Commission - 0.1% (0.001) for both buy and sell
    cerebro.broker.setcommission(commission=0.001)

    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Add observers explicitly since stdstats=False
    # barplot=True makes signals appear above/below the bars, making them more visible
    cerebro.addobservermulti(bt.observers.BuySell, barplot=True)
    cerebro.addobserver(bt.observers.Broker)
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')

    # Apply Strategy
    cerebro.addstrategy(PairStrategyBT)

    results = cerebro.run()
    strat = results[0]
    pyfoliozer = strat.analyzers.getbyname('pyfolio')
    returns, positions, transactions, gross_lev = pyfoliozer.get_pf_items()

    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Plot the result with proper date formatting
    # dayfmt sets the x-axis label format, and numfigs=1 creates a single figure
    cerebro.plot(style='line', volume=False, dayfmt='%Y-%m', start=start_time.date(), end=end_time.date())

    
    