import backtrader as bt

class PlotLineIndicator(bt.Indicator):
    lines = ('line',)
    plotinfo = dict(subplot=False)  # gleiche Achse wie Preis

    def __init__(self):
        # nichts zu fitten, Linie wird von außen gesetzt
        pass