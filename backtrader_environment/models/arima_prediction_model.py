from .prediction_models import PredictionModel
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
import numpy as np
import warnings


class ARIMAModel(PredictionModel):
    """ARIMA model that predicts on log-returns and outputs price forecasts.

    Uses a momentum factor to amplify predictions and generate more trading signals.
    """

    def __init__(self, order=(2, 1, 1), window=252, refit_interval=10, momentum_factor=3.0):
        self.order = order
        self.window = window
        self.refit_interval = refit_interval
        self.momentum_factor = momentum_factor
        self.model_fit: ARIMAResults | None = None
        self._prices = None
        self._log_prices = None
        self._update_counter = 0
        self.requires_refit = False

    def update(self, new_price):
        """Add new price and update model."""
        if self._prices is None:
            return

        self._prices = np.append(self._prices, new_price)
        self._log_prices = np.log(self._prices)
        self._update_counter += 1

        if len(self._prices) > self.window:
            self._prices = self._prices[-self.window:]
            self._log_prices = self._log_prices[-self.window:]

        if self._update_counter % self.refit_interval == 0:
            self._fit_internal()

    def predict(self, n=1):
        if self.model_fit is None or self._prices is None or len(self._prices) == 0:
            return np.nan

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                forecast = self.model_fit.forecast(steps=n)

            last_log_price = self._log_prices[-1]
            forecast_val = forecast.iloc[-1] if hasattr(forecast, "iloc") else forecast[-1]

            amplified_forecast = forecast_val * self.momentum_factor

            predicted_log_price = last_log_price + amplified_forecast
            predicted_price = np.exp(predicted_log_price)

            current_price = self._prices[-1]
            max_change = 0.10 * n
            predicted_price = np.clip(
                predicted_price,
                current_price * (1 - max_change),
                current_price * (1 + max_change)
            )

            return predicted_price
        except Exception:
            return np.nan

    def fit(self, data, window=None):
        if isinstance(data, pd.DataFrame):
            if "close" in data.columns:
                prices = data["close"].values
            elif "Close" in data.columns:
                prices = data["Close"].values
            else:
                prices = data.iloc[:, 0].values
        elif isinstance(data, pd.Series):
            prices = data.values
        else:
            prices = np.array(data, dtype=float)

        prices = prices[~np.isnan(prices)]

        fit_window = self.window if window is None else window
        if len(prices) > fit_window:
            prices = prices[-fit_window:]

        self._prices = prices.copy()
        self._log_prices = np.log(prices)
        return self._fit_internal()

    def _fit_internal(self):
        if self._log_prices is None or len(self._log_prices) < 50:
            return False

        log_returns = np.diff(self._log_prices)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ARIMA(log_returns, order=self.order)
                self.model_fit = model.fit()
            return True
        except Exception:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = ARIMA(log_returns, order=(1, 0, 1))
                    self.model_fit = model.fit()
                return True
            except Exception:
                self.model_fit = None
                return False

    def append(self, data) -> bool:
        """Legacy method for compatibility."""
        if isinstance(data, (int, float)):
            self.update(data)
            return True
        return False
