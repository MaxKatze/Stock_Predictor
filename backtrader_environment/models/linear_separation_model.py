"""LinearSeparation Model: HP-Filter trend decomposition + ARMA residual forecasting."""

import numpy as np
import pandas as pd
import warnings
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

from .prediction_models import PredictionModel


class LinearSeparationModel(PredictionModel):
    """Decomposes log-price into long-term trend, short-term trend, and cyclical component.

    p_t = tau_lang_t + tau_kurz_t + c_t + epsilon_t

    Trends are extrapolated via OLS, cyclical component via ARMA.

    Usage:
        1. Call fit_order() on training data to determine optimal ARMA order via AIC
        2. Call fit() on each bar during walk-forward (HP-Filter + OLS always,
           ARMA only every arma_refit_interval bars)
    """

    def __init__(self, lambda_lang=1600 * 252**2, lambda_kurz=1600,
                 w_lang=252, w_kurz=40, max_p=5, max_q=5, arma_refit_interval=50):
        self.lambda_lang = lambda_lang
        self.lambda_kurz = lambda_kurz
        self.w_lang = w_lang
        self.w_kurz = w_kurz
        self.max_p = max_p
        self.max_q = max_q
        self.arma_refit_interval = arma_refit_interval

        self._slope_lang = None
        self._intercept_lang = None
        self._slope_kurz = None
        self._intercept_kurz = None
        self._arma_model = None
        self._arma_order = None
        self._last_t = None
        self._fitted = False
        self._fit_counter = 0

        self.data_buffer = None
        self.last_close = None
        self.requires_refit = True

    def fit_order(self, data):
        """Determine optimal ARMA order on training data via AIC grid search.

        This should be called once on training data before walk-forward testing.

        Args:
            data: array-like of close prices (training data)
        """
        if isinstance(data, pd.DataFrame):
            prices = data["close"].values if "close" in data.columns else data["Close"].values
        elif isinstance(data, pd.Series):
            prices = data.values
        else:
            prices = np.array(data, dtype=float)

        prices = prices[~np.isnan(prices)]

        if len(prices) < self.w_lang + 50:
            self._arma_order = (1, 1)
            return

        log_prices = np.log(prices)

        tau_lang, _ = hpfilter(log_prices, lamb=self.lambda_lang)
        residual_1 = log_prices - tau_lang
        tau_kurz, _ = hpfilter(residual_1, lamb=self.lambda_kurz)
        c_t = residual_1 - tau_kurz

        best_aic = np.inf
        best_order = (1, 1)

        for p in range(self.max_p + 1):
            for q in range(self.max_q + 1):
                if p == 0 and q == 0:
                    continue
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model = ARIMA(c_t, order=(p, 0, q))
                        result = model.fit(method_kwargs={"maxiter": 200})
                    if result.aic < best_aic:
                        best_aic = result.aic
                        best_order = (p, q)
                except Exception:
                    continue

        self._arma_order = best_order

    def fit(self, data, window=None):
        """Fit the model on historical close prices.

        Args:
            data: array-like of close prices, or DataFrame with 'close' column
            window: ignored (uses all data)
        """
        if isinstance(data, pd.DataFrame):
            prices = data["close"].values if "close" in data.columns else data["Close"].values
        elif isinstance(data, pd.Series):
            prices = data.values
        else:
            prices = np.array(data, dtype=float)

        prices = prices[~np.isnan(prices)]

        if len(prices) < self.w_lang + 50:
            self._fitted = False
            return

        self.data_buffer = prices
        self.last_close = prices[-1]

        log_prices = np.log(prices)
        T = len(log_prices)
        self._last_t = T

        tau_lang, _ = hpfilter(log_prices, lamb=self.lambda_lang)

        self._slope_lang, self._intercept_lang = self._fit_ols_trend(
            tau_lang, self.w_lang
        )

        residual_1 = log_prices - tau_lang

        tau_kurz, _ = hpfilter(residual_1, lamb=self.lambda_kurz)

        self._slope_kurz, self._intercept_kurz = self._fit_ols_trend(
            tau_kurz, self.w_kurz
        )

        c_t = residual_1 - tau_kurz

        self._fit_counter += 1
        if self._arma_model is None or self._fit_counter % self.arma_refit_interval == 0:
            adf_result = adfuller(c_t, autolag="AIC")
            if adf_result[1] >= 0.05:
                warnings.warn(
                    f"Residuals may not be stationary (ADF p={adf_result[1]:.4f}). "
                    "Consider adjusting lambda parameters."
                )
            self._fit_arma(c_t)
        else:
            self._append_arma(c_t[-1])

        self._fitted = True

    def predict(self, n=1):
        """Predict n steps ahead. Returns predicted price(s)."""
        if not self._fitted:
            return float("nan") if n == 1 else np.full(n, float("nan"))

        predictions = []
        for h in range(1, n + 1):
            t_future = self._last_t + h

            tau_lang_hat = self._intercept_lang + self._slope_lang * t_future
            tau_kurz_hat = self._intercept_kurz + self._slope_kurz * t_future

            if self._arma_model is not None:
                try:
                    c_hat = self._arma_model.forecast(steps=h)
                    c_val = c_hat.iloc[-1] if hasattr(c_hat, "iloc") else c_hat[-1]
                except Exception:
                    c_val = 0.0
            else:
                c_val = 0.0

            log_price_hat = tau_lang_hat + tau_kurz_hat + c_val
            predictions.append(np.exp(log_price_hat))

        if n == 1:
            return predictions[0]
        return np.array(predictions)

    def _fit_ols_trend(self, trend_series, window):
        """Fit OLS linear regression on the last `window` points of trend."""
        series = np.array(trend_series)
        n = len(series)
        w = min(window, n)

        t_vals = np.arange(n - w, n, dtype=float)
        y_vals = series[-w:]

        t_mean = t_vals.mean()
        y_mean = y_vals.mean()

        numerator = np.sum((t_vals - t_mean) * (y_vals - y_mean))
        denominator = np.sum((t_vals - t_mean) ** 2)

        if denominator == 0:
            return 0.0, y_mean

        slope = numerator / denominator
        intercept = y_mean - slope * t_mean
        return slope, intercept

    def _fit_arma(self, residuals):
        """Fit ARMA model on residuals using pre-determined order.

        The order must be set via fit_order() on training data first.
        If no order is set, defaults to ARMA(1,1).
        """
        if self._arma_order is None:
            self._arma_order = (1, 1)

        p, q = self._arma_order

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ARIMA(residuals, order=(p, 0, q))
                self._arma_model = model.fit(method_kwargs={"maxiter": 200})
        except Exception:
            self._arma_model = None

    def _append_arma(self, new_residual):
        """Append new residual to ARMA model without refitting parameters.

        Uses statsmodels append() to update the model state for forecasting.
        """
        if self._arma_model is None:
            return

        try:
            self._arma_model = self._arma_model.append([new_residual], refit=False)
        except Exception:
            pass
