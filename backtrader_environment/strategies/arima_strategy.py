import backtrader as bt
from datetime import date
import pandas as pd

from visualization import PlotLineIndicator
from strategies import GeneralStrategy
from models import ARIMAModel

class ARIMAStrategy(GeneralStrategy):
    params = dict(
        arima_order=(1, 1, 1),
        arima_window=100,
        fit_interval=1,  # 0=einmalig, 1=jeden Bar, 2=jeden 2. Bar, etc.
    )

    def __init__(self):
        super().__init__()
        self.models = dict()
        self.forecast_lines = dict()
        self._fit_counters = dict()  # Zähler pro Datenfeed

        for d in self.datas:
            self.models[d] = ARIMAModel(order=self.p.arima_order, window=self.p.arima_window)
            self._fit_counters[d] = 0  # start counter

            # Forecast Linie vorbereiten
            self.forecast_lines[d] = PlotLineIndicator()
            self.forecast_lines[d].plotinfo.plotname = f"{d._name} ARIMA"

    def next(self):
        for d in self.datas:
            closes = pd.Series(d.close.get(size=self.p.arima_window + 20))
            if len(closes) < self.p.arima_window:
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

            # Counter hochzählen
            self._fit_counters[d] += 1

            # 1-step Forecast
            prediction = model.predict(n=1)

            # Linie für Plot setzen
            self.forecast_lines[d].lines.line[0] = prediction
            
            current_close = d.close[0]

            threshold = 0.001
            cash = self.broker.get_cash() * 0.95
            stockprice = current_close
            size = int(cash / stockprice)
            
            if not self.getposition(d).size:
                if prediction >= current_close * (1 + threshold):
                    self.buy(data=d, size=size)
            else:
                if prediction <= current_close * (1 - threshold):
                    self.close(data=d)
