import backtrader as bt
import pandas as pd

from visualization import PlotLineIndicator
from strategies import GeneralStrategy
from models import ARIMAModel
from analyzer import MeanAbsoluteErrorAnalyzer, RootMeanSquaredErrorAnalyzer, MeanSquaredErrorAnalyzer

class ARIMAStrategy(GeneralStrategy):
    params = dict(
        arima_order=(1, 1, 1),
        arima_window=100,
        fit_interval=100,
        signal_threshold=0.001,
    )
    supported_analyzers = [MeanAbsoluteErrorAnalyzer, RootMeanSquaredErrorAnalyzer, MeanSquaredErrorAnalyzer]


    def __init__(self):
        super().__init__()
        self.models = {}
        self.forecast_lines = {}
        self._fit_counters = {}
        self.current_price = {}
        self.current_prediction = {}
        self.predictions = {}
        self.prices = {}
        self.day = 0

        self.days = PlotLineIndicator()
        self.days.plotinfo.plotname = "Days"

        for d in self.datas:
            self.models[d] = ARIMAModel(order=self.p.arima_order, window=self.p.arima_window)
            self._fit_counters[d] = 0
            self.current_price[d] = float("nan")
            self.current_prediction[d] = None
            self.predictions[d] = []
            self.prices[d] = []

            self.forecast_lines[d] = PlotLineIndicator()
            self.forecast_lines[d].plotinfo.plotname = f"{d._name} ARIMA"

    def next(self):
        self.day += 1
        self.days.lines.line[0] = self.day
        for d in self.datas:
            self.prices[d].append(d.close[0])

            closes = pd.Series(d.close.get(size=self.p.arima_window + 20))
            
            if len(closes) < self.p.arima_window:
                self.current_prediction[d] = None
                self.predictions[d].append(self.current_prediction[d])
                self.forecast_lines[d].lines.line[0] = float("nan")
                continue

            model = self.models[d]

            do_fit = False
            if self.p.fit_interval == 0:
                if self._fit_counters[d] == 0:
                    do_fit = True
            else:
                if self._fit_counters[d] % self.p.fit_interval == 0:
                    do_fit = True

            if do_fit:
                model.fit(closes, window=self.p.arima_window)

            self._fit_counters[d] += 1

            if self._fit_counters[d] > 1 and model.model_fit is not None:
                model.append([d.close[0]])

            prediction = model.predict(n=1)
            if pd.isna(prediction):
                self.current_prediction[d] = None
                self.predictions[d].append(self.current_prediction[d])
                self.forecast_lines[d].lines.line[0] = float("nan")
                self.current_price[d] = d.close[0]
                continue

            self.current_prediction[d] = prediction
            self.predictions[d].append(self.current_prediction[d])

            self.forecast_lines[d].lines.line[0] = prediction
            self.current_price[d] = d.close[0]

            if not self.getposition(d).size:
                if prediction >= self.current_price[d] * (1 + self.p.signal_threshold):
                    self.buy(data=d)
            else:
                if prediction <= self.current_price[d] * (1 - self.p.signal_threshold):
                    self.close(data=d)
