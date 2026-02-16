import backtrader as bt

class GeneralStrategy(bt.Strategy):
    def __init__(self):
        self.last_trade = 0
        self.trades = []
    def notify_order(self, order):
        data = order.data
        name = data._name
        if order.status in [order.Completed]:
            price = order.executed.price
            size = order.executed.size
            comm = order.executed.comm
            date = bt.num2date(order.executed.dt)
            sales_type = "SELL" if size < 0 else "BUY"
            print(f"EXEC: {sales_type}: name={name}, price={price}, size={size}, comm={comm} date={date}")
            if size > 0:
                self.last_trade = abs(size) * price
            else:
                trade_return = (abs(size) * price) - self.last_trade
                print(f"TRADE: made {trade_return}")
                self.trades.append(trade_return)
        elif order.status in [order.Canceled]:
            print(f"Order {order.ref} {name} Canceled")
        elif order.status in [order.Margin]:
            print(f"Order {order.ref} {name} Rejected - Margin")
        elif order.status in [order.Rejected]:
            print(f"Order {order.ref} {name} Rejected")