import backtrader as bt

class GeneralStrategy(bt.Strategy):
    def __init__(self):
        self.last_trade = 0
        self.trades = []
        self.analyzer = []