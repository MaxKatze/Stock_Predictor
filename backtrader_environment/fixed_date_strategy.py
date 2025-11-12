import backtrader as bt
from datetime import date

from general_strategy import GeneralStrategy

class FixedDateStrategy(GeneralStrategy):
    params = (
        ("buy_date", date(2000, 1, 5)),
        ("sell_date", date(2000, 11, 20)),
    )

    def next(self):
        for data in self.datas:
            if data._name != "MSFT":
                continue
        
            current_date = data.datetime.date(0)
            
            # Kaufen am Buy-Datum
            if current_date == self.params.buy_date:
                if not self.getposition(data):
                    self.buy(exectype=bt.Order.Market, data=data, size=1)
                    print(f"Kauf am {current_date} zum Open Preis des nächsten Tages: {self.data.open[1]}")
                            
            # Verkaufen am Sell-Datum
            elif current_date == self.params.sell_date:
                if self.getposition(data):
                    self.sell(exectype=bt.Order.Market, data=data, size=1)
                    print(f"Verkauf am {current_date} zum Open Preis des nächsten Tages {self.data.open[1]}")