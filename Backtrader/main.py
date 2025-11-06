from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import backtrader as bt
import os
import sys
import datetime
from FixedDateStrategy import FixedDateStrategy
import pandas
from fileFormat import PandasData

if __name__ == '__main__':

    # Create a cerebro entity
    cerebro = bt.Cerebro(stdstats=False)

    # Add a strategy
    cerebro.addstrategy(FixedDateStrategy)

    # Get a pandas dataframe
    datapath = ('data/assets/MSFT.csv')

    # Simulate the header row isn't there if noheaders requested
    skiprows = 0
    header = 0

    dataframe = pandas.read_csv(datapath,
                                skiprows=skiprows,
                                header=header,
                                parse_dates=True,
                                index_col=0)


    # Pass it to the backtrader datafeed and add it to the cerebro
    data = bt.feeds.PandasData(dataname=dataframe)

    cerebro.adddata(data)

    cerebro.broker.set_cash(100000.0)
    cerebro.broker.setcommission(commission=0.0)

    print("--- Starting Portfolio Value:", cerebro.broker.getvalue(), " ---")

    # Run over everything
    cerebro.run()

    print("--- Ending Portfolio Value:", cerebro.broker.getvalue(), " ---")

    # Plot the result
    cerebro.plot(style='bar')