"""LinearSeparation Model: HP-Filter trend decomposition + ARIMA residual forecasting."""

import numpy as np
import pandas as pd
import warnings
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.arima.model import ARIMA

from .prediction_models import PredictionModel


class LinearSeparationModel(PredictionModel):
    """Decomposes log-price into long-term trend, short-term trend, and cyclical component.

    p_t = tau_lang_t + tau_kurz_t + c_t + epsilon_t

    Uses rolling HP-filter decomposition with ARIMA(p,1,q) for the cyclical component.
    Updates predictions daily based on recent price data.
    """

    def __init__(self, lambda_lang=1600 * 252**2, lambda_kurz=1600,
                 w_lang=252, w_kurz=40, arima_order=(2, 1, 1),
                 rolling_window=500, refit_interval=5):
        self.lambda_lang = lambda_lang
        self.lambda_kurz = lambda_kurz
        self.w_lang = w_lang
        self.w_kurz = w_kurz
        self.arima_order = arima_order
        self.rolling_window = rolling_window
        self.refit_interval = refit_interval

        self._arima_model = None
        self._fitted = False
        self._fit_counter = 0

        self._prices = None
        self._tau_lang = None
        self._tau_kurz = None
        self._c_t = None

        self.data_buffer = None
        self.last_close = None
        self.requires_refit = False

    def fit_order(self, data):
        """Compatibility method - order is now fixed via constructor."""
        pass

    def _extract_prices(self, data):
        """Extract price array from various input formats."""
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
        return prices[~np.isnan(prices)]

    def fit(self, data, window=None):
        """Fit the model on historical close prices (training data)."""
        prices = self._extract_prices(data)

        min_required = max(self.w_lang, 100)
        if len(prices) < min_required:
            self._fitted = False
            return

        self._prices = prices.copy()
        self.last_close = prices[-1]
        self._decompose()
        self._fit_arima()
        self._fitted = True

    def update(self, new_price):
        """Add a new price and update the model (walk-forward step)."""
        if self._prices is None:
            return

        self._prices = np.append(self._prices, new_price)
        self.last_close = new_price
        self._fit_counter += 1

        if len(self._prices) > self.rolling_window:
            self._prices = self._prices[-self.rolling_window:]

        self._decompose()

        if self._fit_counter % self.refit_interval == 0:
            self._fit_arima()

    def _decompose(self):
        """Decompose prices using HP filter."""
        if self._prices is None or len(self._prices) < 100:
            return

        log_prices = np.log(self._prices)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._tau_lang, _ = hpfilter(log_prices, lamb=self.lambda_lang)
            residual_1 = log_prices - self._tau_lang
            self._tau_kurz, _ = hpfilter(residual_1, lamb=self.lambda_kurz)
            self._c_t = residual_1 - self._tau_kurz

    def predict(self, n=1):
        """Predict n steps ahead. Returns predicted price(s)."""
        if not self._fitted or self._prices is None or self._tau_lang is None:
            return float("nan") if n == 1 else np.full(n, float("nan"))

        current_log_price = np.log(self._prices[-1])

        slope_lang = self._compute_slope(self._tau_lang, min(self.w_lang, len(self._tau_lang) - 1))
        slope_kurz = self._compute_slope(self._tau_kurz, min(self.w_kurz, len(self._tau_kurz) - 1))

        predictions = []
        for h in range(1, n + 1):
            tau_lang_hat = self._tau_lang[-1] + slope_lang * h
            tau_kurz_hat = self._tau_kurz[-1] + slope_kurz * h

            c_hat = 0.0
            if self._arima_model is not None:
                try:
                    forecast = self._arima_model.forecast(steps=h)
                    c_hat = forecast.iloc[-1] if hasattr(forecast, "iloc") else forecast[-1]
                except Exception:
                    c_hat = self._c_t[-1] if self._c_t is not None and len(self._c_t) > 0 else 0.0

            log_price_hat = tau_lang_hat + tau_kurz_hat + c_hat

            max_daily_change = 0.05
            log_price_hat = np.clip(
                log_price_hat,
                current_log_price - max_daily_change * h,
                current_log_price + max_daily_change * h
            )

            predictions.append(np.exp(log_price_hat))

        if n == 1:
            return predictions[0]
        return np.array(predictions)

    def _compute_slope(self, series, window):
        """Compute slope from recent window using simple linear regression."""
        if series is None or window < 2:
            return 0.0

        y = series[-window:]
        x = np.arange(len(y))

        x_mean = x.mean()
        y_mean = y.mean()

        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def _fit_arima(self):
        """Fit ARIMA model on the cyclical component."""
        if self._c_t is None or len(self._c_t) < 50:
            self._arima_model = None
            return

        p, d, q = self.arima_order

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ARIMA(self._c_t, order=(p, d, q))
                self._arima_model = model.fit(method_kwargs={"maxiter": 200})
        except Exception:
            try:
                model = ARIMA(self._c_t, order=(1, 1, 1))
                self._arima_model = model.fit(method_kwargs={"maxiter": 200})
            except Exception:
                self._arima_model = None
