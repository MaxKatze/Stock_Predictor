import backtrader as bt

from strategies import GeneralStrategy

class MovingAverageStrategy(GeneralStrategy):
    params = (
        ("fast_period", 10),
        ("slow_period", 30),
    )

    def __init__(self):
        super().__init__()
        print("-------Data:---------")
        for i, d in enumerate(self.datas):
            print(i, d._name)  # zeigt alle Datafeeds

        self.fast_mas = dict()
        self.slow_mas = dict()
        self.crossovers = dict()

        for i, d in enumerate(self.datas):
            self.fast_mas[d] = bt.indicators.SimpleMovingAverage(d.close, period=self.params.fast_period)
            self.slow_mas[d] = bt.indicators.SimpleMovingAverage(d.close, period=self.params.slow_period)
            self.crossovers[d] = bt.indicators.CrossOver(self.fast_mas[d], self.slow_mas[d])

    def next(self):
        for d in self.datas:
            crossover = self.crossovers[d][0]
            position = self.getposition(d).size

            if crossover > 0:
                if position == 0:
                    cash = self.broker.get_cash() * 0.95
                    stockprice = d.close[0]
                    size = int(cash / stockprice)
                    if size > 0:
                        dt = d.datetime.date(0)
                        print(f"Should buy on {dt} for {stockprice}")
                        self.buy(data=d, size=size)
            if crossover < 0:
                if position > 0:
                    stockprice = d.close[0]
                    dt = d.datetime.date(0)
                    print(f"Should sell on {dt} for {stockprice}")
                    self.close(data=d)
                    
