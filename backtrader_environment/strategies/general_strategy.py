import backtrader as bt


class GeneralStrategy(bt.Strategy):
    def __init__(self):
        self.last_trade = 0
        self.trades = []

    def notify_order(self, order):
        pass
        # if order.status in [order.Completed]:

        #     if order.isbuy():
        #         print(
        #             f"BUY EXECUTED: Size={order.executed.size} "
        #             f"Price={order.executed.price}"
        #         )

        #     elif order.issell():
        #         print(
        #             f"SELL EXECUTED: Size={order.executed.size} "
        #             f"Price={order.executed.price}")
