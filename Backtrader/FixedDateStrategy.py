import backtrader as bt
from datetime import date

class FixedDateStrategy(bt.Strategy):
    params = (
        ("buy_date", date(2000, 1, 5)),   # hart codiertes Kaufdatum
        ("sell_date", date(2000, 11, 20)),  # hart codiertes Verkaufsdatum
    )

    def next(self):
        # Aktuelles Datum des Datenfeeds
        current_date = self.datas[0].datetime.date(0)
        
        # Kaufen am Buy-Datum
        if current_date == self.params.buy_date:
            if not self.position:  # keine offene Position
                self.buy()
                print(f"Kauf am {current_date} zum Preis {self.data.close[0]}")
        
        # Verkaufen am Sell-Datum
        elif current_date == self.params.sell_date:
            if self.position:  # nur verkaufen, wenn Position offen
                self.sell()
                print(f"Verkauf am {current_date} zum Preis {self.data.close[0]}")