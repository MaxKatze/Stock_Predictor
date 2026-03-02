import backtrader as bt


class PlotLineIndicator(bt.Indicator):
    lines = ('line',)
    plotinfo = dict(subplot=False)

    def __init__(self):
        pass
