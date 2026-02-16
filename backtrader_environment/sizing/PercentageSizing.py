from .GeneralSizing import GeneralSizer

class PercentageSizer(GeneralSizer):
    params = dict(perc=0.95)

    def _getsizing(self, comminfo, cash, data, isbuy):
        if isbuy:
            size = (cash * self.p.perc) / data.close[0]
            return int(size)
        else: #close whole position
            return self.strategy.getposition(data).size