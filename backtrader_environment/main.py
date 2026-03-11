from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import backtrader as bt
import pandas
from strategies import ARIMAStrategy, MovingAverageStrategy 
from analyzer import MeanAbsoluteErrorAnalyzer, RootMeanSquaredErrorAnalyzer 
from sizing import PercentageSizer
from pprint import pprint


def load_csv_feed(symbol: str) -> tuple[bt.feeds.PandasData, float]:
    datapath = f"data/assets/{symbol}.csv"
    dataframe = pandas.read_csv(
        datapath,
        skiprows=0,
        header=0,
        parse_dates=True,
        index_col=0,
    )
    begin_stock_price = dataframe["Close"].iloc[0]
    end_stock_price = dataframe["Close"].iloc[-1]
    percentage_price_increase = end_stock_price / begin_stock_price
    return bt.feeds.PandasData(dataname=dataframe), percentage_price_increase


if __name__ == '__main__':
    cerebro = bt.Cerebro(stdstats=True)
    cerebro.addstrategy(MovingAverageStrategy)
    # cerebro.addanalyzer(MeanAbsoluteErrorAnalyzer, _name="mae")
    # cerebro.addanalyzer(RootMeanSquaredErrorAnalyzer, _name="rmse")
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addsizer(PercentageSizer)

    stocks = ["SAP"]

    average_percentage_stock_increase = 0
    for stock in stocks:
        data, percentage_price = load_csv_feed(stock)
        average_percentage_stock_increase += percentage_price
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

    for analyzer_name in results[0].analyzers.getnames():
        analyzer = results[0].analyzers.getbyname(analyzer_name)
        print(f"{analyzer_name}: {analyzer.get_analysis()}")

    cerebro.plot(style='line')
