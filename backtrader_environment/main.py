from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import backtrader as bt
import pandas
from strategies import MovingAverageStrategy, ARIMAStrategy
from analyzer import MeanAbsoluteErrorAnalyzer, RootMeanSquaredErrorAnalyzer
from sizing import PercentageSizer
if __name__ == '__main__':
    # Create a cerebro entity
    cerebro = bt.Cerebro(stdstats=True)

    # Add a strategy
    cerebro.addstrategy(ARIMAStrategy)

    cerebro.addanalyzer(MeanAbsoluteErrorAnalyzer, _name="mae")
    cerebro.addanalyzer(RootMeanSquaredErrorAnalyzer, _name="rmse")
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    
    cerebro.addsizer(PercentageSizer)

    stocks = ["SAP"] #, "AAPL", "ORCL", "SAP"

    average_percentage_stock_increase = 0
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


        begin_stockprice = dataframe["Close"].iloc[0]
        end_stockprice = dataframe["Close"].iloc[-1]
        precentage_price = end_stockprice / begin_stockprice
        average_percentage_stock_increase += precentage_price


        data = bt.feeds.PandasData(dataname=dataframe)

        cerebro.adddata(data, name=stock)

    average_percentage_stock_increase /= len(stocks)
    cash = 100000.0

    buy_and_hold_ending_value = cash * average_percentage_stock_increase

    print(f"Buy and Hold Ending Portfolio Value: {buy_and_hold_ending_value}")

    cerebro.broker.set_cash(cash)
    cerebro.broker.setcommission(commission=0.0)

    print("--- Starting Portfolio Value:", cerebro.broker.getvalue(), " ---")
    # Run over everything
    results = cerebro.run()
    print("--- Ending Portfolio Value:", cerebro.broker.getvalue(), " ---")

    sharpe = results[0].analyzers.sharpe.get_analysis()
    mae = results[0].analyzers.mae.get_analysis()
    rsme = results[0].analyzers.rmse.get_analysis()

    print(rsme)
    print(sharpe)
    print(mae)

    # Plot the result
    cerebro.plot(style='line')