import backtrader as bt
from datetime import date

class FixedDateStrategy(bt.Strategy):
    params = (
        ("buy_date", date(2000, 1, 5)),   # hart codiertes Kaufdatum
        ("after_date", date(2000, 1, 6)), # day after buy
        ("sell_date", date(2000, 11, 20)),  # hart codiertes Verkaufsdatum
    )

    def next(self):
        # Aktuelles Datum des Datenfeeds
        current_date = self.datas[0].datetime.date(0)
        
        # Kaufen am Buy-Datum
        if current_date == self.params.buy_date:
            if not self.position:  # keine offene Position
                self.buy(exectype=bt.Order.Market)
                print(f"Kauf am {current_date} zum Open Preis des nächsten Tages: {self.data.open[1]}")
                        
        # Verkaufen am Sell-Datum
        elif current_date == self.params.sell_date:
            if self.position:  # nur verkaufen, wenn Position offen
                self.sell(exectype=bt.Order.Market)
                print(f"Verkauf am {current_date} zum Preis {self.data.open[1]}")
    
    def notify_order(self, order):
        if order.status in [order.Completed]:
            price = order.executed.price
            size = order.executed.size
            comm = order.executed.comm
            print(f"Executed: price={price}, size={size}, comm={comm}")
            print(f"Cash now: {self.broker.getcash():.8f}")