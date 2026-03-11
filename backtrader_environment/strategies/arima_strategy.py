import backtrader as bt
import pandas as pd

from visualization import PlotLineIndicator
from strategies import GeneralStrategy
from models import ARIMAModel
from analyzer import MeanAbsoluteErrorAnalyzer, RootMeanSquaredErrorAnalyzer, MeanSquaredErrorAnalyzer

class ARIMAStrategy(GeneralStrategy):
    params = dict(
        arima_order=(2, 1, 1),
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
        self._next_plot_prediction = {}
        self.days = 0
        self.prediction_today_difference = 0

        for d in self.datas:
            self.models[d] = ARIMAModel(order=self.p.arima_order, window=self.p.arima_window)
            self._fit_counters[d] = 0
            self.current_price[d] = float("nan")
            self.current_prediction[d] = None
            self.predictions[d] = []
            self.prices[d] = []
            self._next_plot_prediction[d] = None

            self.forecast_lines[d] = PlotLineIndicator()
            self.forecast_lines[d].plotinfo.plotname = f"{d._name} ARIMA"

    def next(self):
        self.days += 1
        for d in self.datas:
            delayed_prediction = self._next_plot_prediction[d]
            self.forecast_lines[d].lines.line[0] = (
                float("nan") if delayed_prediction is None else delayed_prediction
            )

            self.prices[d].append(d.close[0])

            closes = pd.Series(d.close.get(size=self.p.arima_window + 20))
            
            if len(closes) < self.p.arima_window:
                self.current_prediction[d] = None
                self.predictions[d].append(self.current_prediction[d])
                self._next_plot_prediction[d] = None
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

            if (not do_fit) and self._fit_counters[d] > 1 and model.model_fit is not None:
                model.append([d.close[0]])

            prediction = model.predict(n=1)
            if pd.isna(prediction):
                self.current_prediction[d] = None
                self.predictions[d].append(self.current_prediction[d])
                self._next_plot_prediction[d] = None
                self.current_price[d] = d.close[0]
                continue
            
            self.current_prediction[d] = prediction
            self.predictions[d].append(self.current_prediction[d])
            self._next_plot_prediction[d] = prediction
            self.current_price[d] = d.close[0]
            self.prediction_today_difference += (abs(prediction - self.current_price[d]))

            if not self.getposition(d).size:
                if prediction >= self.current_price[d] * (1 + self.p.signal_threshold):
                    self.buy(data=d)
            else:
                if prediction <= self.current_price[d] * (1 - self.p.signal_threshold):
                    self.close(data=d)
    
    def stop(self):
        print("stopped")
        avg_price_prediction_diff = self.prediction_today_difference / self.days
        print(f"average price-prediction difference: {avg_price_prediction_diff}")