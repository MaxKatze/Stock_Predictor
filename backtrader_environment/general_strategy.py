import backtrader as bt

class GeneralStrategy(bt.Strategy):
    def notify_order(self, order):
        data = order.data
        name = data._name
        if order.status in [order.Submitted, order.Accepted]:
            print(f"Order {order.ref} {name} Submitted/Accepted")
            return
        if order.status in [order.Completed]:
            price = order.executed.price
            size = order.executed.size
            comm = order.executed.comm
            date = bt.num2date(order.executed.dt)
            print(f"Executed: name={name}, price={price}, size={size}, comm={comm} date={date}")
        elif order.status in [order.Canceled]:
            print(f"Order {order.ref} {name} Canceled")
        elif order.status in [order.Margin]:
            print(f"Order {order.ref} {name} Rejected - Margin")
        elif order.status in [order.Rejected]:
            print(f"Order {order.ref} {name} Rejected")