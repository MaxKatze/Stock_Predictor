import backtrader as bt
from datetime import date

from strategies import GeneralStrategy


class FixedDateStrategy(GeneralStrategy):
    params = (
        ("buy_date", date(2000, 1, 5)),
        ("sell_date", date(2000, 11, 20)),
    )

    def __init__(self):
        super().__init__()

    def next(self):
        for data in self.datas:
            if data._name != "MSFT":
                continue
        
            current_date = data.datetime.date(0)
            
            if current_date == self.params.buy_date:
                if not self.getposition(data):
                    self.buy(exectype=bt.Order.Market, data=data, size=1)
                    print(f"Buy on {current_date}; next session open: {data.open[1]}")
                            
            elif current_date == self.params.sell_date:
                if self.getposition(data):
                    self.sell(exectype=bt.Order.Market, data=data, size=1)
                    print(f"Sell on {current_date}; next session open: {data.open[1]}")
