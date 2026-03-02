import backtrader as bt
from datetime import date
import pandas as pd

from visualization import PlotLineIndicator
from strategies import GeneralStrategy
from models import ARIMAModel
from analyzer import MeanAbsoluteErrorAnalyzer, RootMeanSquaredErrorAnalyzer, MeanSquaredErrorAnalyzer

class ARIMAStrategy(GeneralStrategy):
    params = dict(
        arima_order=(1, 1, 1),
        arima_window=100,
        fit_interval=100,  # 0=einmalig, 1=jeden Bar, 2=jeden 2. Bar, etc.
    )
    supported_analyzers = [MeanAbsoluteErrorAnalyzer, RootMeanSquaredErrorAnalyzer, MeanSquaredErrorAnalyzer]


    def __init__(self):
        super().__init__()
        self.models = dict()
        self.forecast_lines = dict()
        self._fit_counters = dict()  # Zähler pro Datenfeed
        self.current_price = {}
        self.current_prediction = {}
        self.predictions = {}
        self.prices = {}


        for d in self.datas:
            self.models[d] = ARIMAModel(order=self.p.arima_order, window=self.p.arima_window)
            self._fit_counters[d] = 0  # start counter
            self.current_price[d] = -1
            self.current_prediction[d] = -1
            self.predictions[d] = []
            self.prices[d] = []

            # Forecast Linie vorbereiten
            self.days = PlotLineIndicator()
            self.day = 0
            self.days.plotinfo.plotname = "Days"
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
                continue  # nicht genug Daten

            model = self.models[d]
            
            # Fit-Bedingung prüfen
            do_fit = False
            if self.p.fit_interval == 0:
                # einmalig beim ersten Mal
                if self._fit_counters[d] == 0:
                    do_fit = True
            else:
                # Rolling Fit alle 'fit_interval' Bars
                if self._fit_counters[d] % self.p.fit_interval == 0:
                    do_fit = True

            if do_fit:
                model.fit(closes, window=self.p.arima_window)

            model.append([d.close[0]])
            # Counter hochzählen
            self._fit_counters[d] += 1

            # 1-step Forecast
            prediction = model.predict(n=1)
            self.current_prediction[d] = prediction
            self.predictions[d].append(self.current_prediction[d])

            # Linie für Plot setzen
            self.forecast_lines[d].lines.line[0] = prediction
            
            self.current_price[d] = d.close[0] #USE JUST CLOSE PRICES

            threshold = 0.001
            cash = self.broker.get_cash() * 0.95
            size = int(cash / self.current_price[d])
            
            if not self.getposition(d).size:
                if prediction >= self.current_price[d] * (1 + threshold):
                    self.buy(data=d)
            else:
                if prediction <= self.current_price[d] * (1 - threshold):
                    self.close(data=d)
