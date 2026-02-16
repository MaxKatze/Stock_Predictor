from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import backtrader as bt
import os
import sys
import datetime
from strategies.fixed_date_strategy import FixedDateStrategy
import pandas
from data_handling import PandasData
from strategies.moving_average_strategy import MovingAverageStrategy
from strategies.arima_strategy import ARIMAStrategy

if __name__ == '__main__':

    # Create a cerebro entity
    cerebro = bt.Cerebro(stdstats=False)

    # Add a strategy
    cerebro.addstrategy(MovingAverageStrategy)

    stocks = ["SAP"] #, "AAPL", "ORCL", "SAP"

    for stock in stocks:
        # Get a pandas dataframe
        datapath = (f"data/assets/{stock}.csv")

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

        cerebro.adddata(data, name=stock)

    cerebro.broker.set_cash(100000.0)
    cerebro.broker.setcommission(commission=0.0)

    print("--- Starting Portfolio Value:", cerebro.broker.getvalue(), " ---")
    # Run over everything
    cerebro.run()
    print("--- Ending Portfolio Value:", cerebro.broker.getvalue(), " ---")

    # Plot the result
    cerebro.plot(style='bar')