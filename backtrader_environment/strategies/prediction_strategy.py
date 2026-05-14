"""Prediction Strategy as specified in Konzeption Section 3.4.

Implements the trading logic:
- Forecasting Horizon k=1 (daily predictions)
- Signal based on predicted return r̂_{t+1} and threshold ε:
    - r̂_{t+1} > ε      → Long (adjust position to Kelly target)
    - -ε ≤ r̂_{t+1} ≤ ε → Hold (no action)
    - r̂_{t+1} < -ε     → Cash (close all positions)
- ε (threshold) = MAE on validation data
- Position sizing: Half-Kelly criterion with position rebalancing
"""

import backtrader as bt
import pandas as pd
import numpy as np

from strategies import GeneralStrategy
from sizing.HalfKellySizer import HalfKellyPositionManager
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
    """Trading strategy based on model predictions following Konzeption Section 3.4.

    Signal logic (Section 3.4.2):
        Signal_t = Long  if r̂_{t+1} > ε
        Signal_t = Hold  if -ε ≤ r̂_{t+1} ≤ ε
        Signal_t = Cash  if r̂_{t+1} < -ε

    Position sizing (Section 3.4.3):
        When Signal = Long: adjust position to f_t * portfolio_value
        When Signal = Hold: no action (keep current position)
        When Signal = Cash: close ALL positions

    Where:
        ε = signal_threshold (MAE on validation data)
        f_t = Half-Kelly fraction = 0.5 * μ_t / σ²_t
    """

    params = dict(
        model=None,
        signal_threshold=0.01,
        forecast_horizon=1,
        warmup_period=60,
        fit_interval=0,
        kelly_window=60,
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

        self.kelly_manager = HalfKellyPositionManager(
            kelly_window=self.p.kelly_window,
            min_observations=30,
        )

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
                actual_return = np.log(current_price / past_price)
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

            df = self._get_ohlcv_dataframe(d)

            if hasattr(model, "requires_refit") and model.requires_refit:
                model.fit(df)
                self._model_fitted[d] = True
            elif self.p.fit_interval > 0 and self._fit_counters[d] % self.p.fit_interval == 0:
                model.fit(df)
                self._model_fitted[d] = True
            elif hasattr(model, "data_buffer"):
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
                self._handle_cash_signal(d)
                continue

            self.current_prediction[d] = prediction
            self.predictions[d].append(prediction)

            predicted_return = np.log(prediction / current_price)
            self.predicted_returns[d].append(predicted_return)

            if predicted_return > self.p.signal_threshold:
                self._handle_long_signal(d)
            elif predicted_return < -self.p.signal_threshold:
                self._handle_cash_signal(d)
            # else: -ε ≤ predicted_return ≤ ε → Hold (no action)

    def _handle_long_signal(self, d):
        """Handle Long signal: adjust position to Half-Kelly target.

        From Konzeption Section 3.4.3:
        "Ist also zum Zeitpunkt t Signal_t = Long, wird die Position so angepasst,
        dass sie dem Anteil f_t vom Gesamtportfoliowert entspricht.
        Dies kann auch einer Reduktion der Position entsprechen."
        """
        current_position = self.getposition(d).size
        current_price = d.close[0]
        portfolio_value = self.broker.getvalue()
        available_cash = self.broker.getcash()

        order_size = self.kelly_manager.compute_order_size(
            data=d,
            portfolio_value=portfolio_value,
            current_position=current_position,
            current_price=current_price,
            available_cash=available_cash,
        )

        if order_size > 0:
            self.buy(data=d, size=order_size)
        elif order_size < 0:
            self.sell(data=d, size=abs(order_size))

    def _handle_cash_signal(self, d):
        """Handle Cash signal: close ALL positions."""
        if self.getposition(d).size:
            self.close(data=d)

    def stop(self):
        total_predictions = sum(
            1 for d in self.datas
            for p in self.predictions[d] if p is not None
        )
        if total_predictions > 0:
            print(f"PredictionStrategy: {total_predictions} predictions made across {len(self.datas)} feeds.")
