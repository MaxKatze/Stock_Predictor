"""Preprocessed SVR Model: Support Vector Regression with technical indicator features."""

import numpy as np
import pandas as pd
import warnings
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from .prediction_models import PredictionModel


class SVRPredictionModel(PredictionModel):
    """SVR model predicting log-returns using preprocessed technical features.

    Uses RBF kernel with grid search for hyperparameter tuning on validation data.
    Features are the same 14 technical indicators used by the LSTM model.
    """

    def __init__(self, C=1.0, epsilon=0.1, gamma=1.0, lookback_window=60):
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.lookback_window = lookback_window

        self.svr = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.data_buffer = None
        self.last_close = None

    def fit(self, data, window=None, validation_data=None):
        """Fit SVR on training data. Optionally tune hyperparameters on validation data.

        Args:
            data: DataFrame with OHLCV columns (training data)
            window: ignored
            validation_data: optional DataFrame for grid search tuning
        """
        df = self._to_dataframe(data)
        if len(df) < self.lookback_window + 60:
            self.is_fitted = False
            return

        self.data_buffer = df.copy()
        self.last_close = df["close"].iloc[-1]

        features_df = _compute_features(df)
        targets = np.log(df["close"] / df["close"].shift(1)).values

        valid_mask = ~np.isnan(targets)
        features_arr = features_df.values[valid_mask]
        targets_arr = targets[valid_mask]

        targets_arr = np.clip(targets_arr, -0.2, 0.2)

        self.scaler.fit(features_arr)
        X_train = self.scaler.transform(features_arr)
        y_train = targets_arr

        if validation_data is not None:
            self._grid_search(X_train, y_train, validation_data)
        else:
            self.svr = SVR(kernel="rbf", C=self.C, epsilon=self.epsilon, gamma=self.gamma)
            self.svr.fit(X_train, y_train)

        self.is_fitted = True

    def predict(self, n=1):
        """Predict next price based on current features."""
        if not self.is_fitted or self.data_buffer is None:
            return float("nan")

        features_df = _compute_features(self.data_buffer)
        last_features = features_df.values[-1:]
        X = self.scaler.transform(last_features)

        r_hat = self.svr.predict(X)[0]
        price_hat = self.last_close * np.exp(r_hat)

        if n == 1:
            return price_hat
        return np.array([price_hat] * n)

    def predict_return(self):
        """Predict log-return directly."""
        if not self.is_fitted or self.data_buffer is None:
            return float("nan")

        features_df = _compute_features(self.data_buffer)
        last_features = features_df.values[-1:]
        X = self.scaler.transform(last_features)
        return self.svr.predict(X)[0]

    def append(self, new_data):
        """Append new OHLCV row to data buffer (for walk-forward)."""
        if self.data_buffer is not None:
            new_df = self._to_dataframe(new_data)
            self.data_buffer = pd.concat([self.data_buffer, new_df])
            self.last_close = self.data_buffer["close"].iloc[-1]

    def _grid_search(self, X_train, y_train, validation_data):
        """Grid search over C, epsilon, gamma on validation set."""
        val_df = self._to_dataframe(validation_data)
        val_features = _compute_features(val_df)
        val_targets = np.log(val_df["close"] / val_df["close"].shift(1)).values

        valid_mask = ~np.isnan(val_targets)
        X_val = self.scaler.transform(val_features.values[valid_mask])
        y_val = val_targets[valid_mask]
        y_val = np.clip(y_val, -0.2, 0.2)

        C_grid = [0.1, 1, 10, 100, 1000]
        eps_grid = [0.001, 0.01, 0.1, 0.5, 1]
        gamma_grid = [0.01, 0.1, 1, 10, 50, 100]

        best_rmse = np.inf
        best_params = (self.C, self.epsilon, self.gamma)

        for C in C_grid:
            for eps in eps_grid:
                for gamma in gamma_grid:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            svr = SVR(kernel="rbf", C=C, epsilon=eps, gamma=gamma)
                            svr.fit(X_train, y_train)
                            y_pred = svr.predict(X_val)
                            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                            if rmse < best_rmse:
                                best_rmse = rmse
                                best_params = (C, eps, gamma)
                    except Exception:
                        continue

        self.C, self.epsilon, self.gamma = best_params
        self.svr = SVR(kernel="rbf", C=self.C, epsilon=self.epsilon, gamma=self.gamma)
        self.svr.fit(X_train, y_train)

    def _to_dataframe(self, data):
        if isinstance(data, pd.Series):
            return pd.DataFrame({"close": data})
        df = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        df.columns = [c.lower() for c in df.columns]
        return df


def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 14 technical indicator features from OHLCV data.

    Same feature set as the LSTM model for consistency.
    """
    features = pd.DataFrame(index=df.index)

    features["return_1d"] = df["close"].pct_change(1)
    features["return_5d"] = df["close"].pct_change(5)
    features["return_10d"] = df["close"].pct_change(10)
    features["return_20d"] = df["close"].pct_change(20)

    if "high" in df.columns and "low" in df.columns:
        features["hl_range"] = (df["high"] - df["low"]) / df["close"]
    else:
        features["hl_range"] = 0

    if "open" in df.columns:
        features["oc_change"] = (df["close"] - df["open"]) / df["open"]
    else:
        features["oc_change"] = 0

    if "volume" in df.columns and df["volume"].sum() > 0:
        features["volume_change"] = np.log1p(df["volume"]).pct_change()
    else:
        features["volume_change"] = 0

    features["volatility_5d"] = df["close"].pct_change().rolling(5).std()
    features["volatility_20d"] = df["close"].pct_change().rolling(20).std()

    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    features["rsi_14"] = (100 - (100 / (1 + rs))) / 100

    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    features["macd_pct"] = macd / df["close"]
    features["macd_signal_pct"] = macd.ewm(span=9, adjust=False).mean() / df["close"]

    sma_20 = df["close"].rolling(20).mean()
    sma_50 = df["close"].rolling(50).mean()
    features["sma_ratio_20"] = df["close"] / (sma_20 + 1e-10) - 1
    features["sma_ratio_50"] = df["close"] / (sma_50 + 1e-10) - 1

    features = features.ffill().fillna(0)
    features = features.replace([np.inf, -np.inf], 0)

    return features
