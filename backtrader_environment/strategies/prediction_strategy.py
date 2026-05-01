"""Generic prediction strategy that works with any PredictionModel."""

import backtrader as bt
import pandas as pd
import numpy as np

from strategies import GeneralStrategy
from analyzer import (
    MeanAbsoluteErrorAnalyzer,
    RootMeanSquaredErrorAnalyzer,
    MeanSquaredErrorAnalyzer,
    RSquaredAnalyzer,
    MAPEAnalyzer,
    OOSRSquaredAnalyzer,
    DirectionalAccuracyAnalyzer,
)


class PredictionStrategy(GeneralStrategy):
    """A unified strategy that works with any PredictionModel subclass.

    The model is passed in via params (pre-trained). This strategy handles:
    - Feeding data to the model
    - Generating buy/sell signals from predictions
    - Tracking predictions/prices for analyzers
    """

    params = dict(
        model=None,
        signal_threshold=0.01,
        forecast_horizon=1,
        warmup_period=60,
        fit_interval=0,
    )

    supported_analyzers = [
        MeanAbsoluteErrorAnalyzer,
        RootMeanSquaredErrorAnalyzer,
        MeanSquaredErrorAnalyzer,
        RSquaredAnalyzer,
        MAPEAnalyzer,
        OOSRSquaredAnalyzer,
        DirectionalAccuracyAnalyzer,
    ]

    def __init__(self):
        super().__init__()

        self.models = {}
        self._fit_counters = {}
        self._model_fitted = {}
        self.current_price = {}
        self.current_prediction = {}
        self.predictions = {}
        self.prices = {}
        self.predicted_returns = {}
        self.actual_returns = {}

        for d in self.datas:
            self.models[d] = self.p.model
            self._model_fitted[d] = self.p.model is not None
            self._fit_counters[d] = 0
            self.current_price[d] = float("nan")
            self.current_prediction[d] = None
            self.predictions[d] = []
            self.prices[d] = []
            self.predicted_returns[d] = []
            self.actual_returns[d] = []

    def _get_ohlcv_dataframe(self, d):
        """Extract OHLCV data from backtrader data feed."""
        size = len(d)
        closes = list(d.close.get(size=size))
        opens = list(d.open.get(size=size)) if hasattr(d, "open") else closes
        highs = list(d.high.get(size=size)) if hasattr(d, "high") else closes
        lows = list(d.low.get(size=size)) if hasattr(d, "low") else closes
        volumes = list(d.volume.get(size=size)) if hasattr(d, "volume") else [0] * len(closes)

        return pd.DataFrame({
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        })

    def next(self):
        for d in self.datas:
            current_price = d.close[0]
            self.prices[d].append(current_price)
            self.current_price[d] = current_price

            horizon = self.p.forecast_horizon
            if len(self.prices[d]) > horizon:
                past_price = self.prices[d][-(horizon + 1)]
                actual_return = (current_price - past_price) / past_price
                self.actual_returns[d].append(actual_return)

            model = self.models[d]
            if model is None:
                self.current_prediction[d] = None
                self.predictions[d].append(None)
                self.predicted_returns[d].append(None)
                continue

            available_bars = len(d)
            if available_bars < self.p.warmup_period:
                self.current_prediction[d] = None
                self.predictions[d].append(None)
                self.predicted_returns[d].append(None)
                continue

            self._fit_counters[d] += 1

            if self.p.fit_interval > 0 and self._fit_counters[d] % self.p.fit_interval == 0:
                df = self._get_ohlcv_dataframe(d)
                model.fit(df)
                self._model_fitted[d] = True

            if hasattr(model, "data_buffer") and model.data_buffer is not None:
                df = self._get_ohlcv_dataframe(d)
                model.data_buffer = df
                model.last_close = current_price

            try:
                prediction = model.predict(n=1)
            except Exception:
                prediction = float("nan")

            if prediction is None or (isinstance(prediction, float) and np.isnan(prediction)):
                self.current_prediction[d] = None
                self.predictions[d].append(None)
                self.predicted_returns[d].append(None)
                continue

            self.current_prediction[d] = prediction
            self.predictions[d].append(prediction)

            predicted_return = (prediction - current_price) / current_price
            self.predicted_returns[d].append(predicted_return)

            if not self.getposition(d).size:
                if predicted_return > self.p.signal_threshold:
                    self.buy(data=d)
            else:
                if predicted_return < -self.p.signal_threshold:
                    self.close(data=d)

    def stop(self):
        total_predictions = sum(
            1 for d in self.datas
            for p in self.predictions[d] if p is not None
        )
        if total_predictions > 0:
            print(f"PredictionStrategy: {total_predictions} predictions made across {len(self.datas)} feeds.")
