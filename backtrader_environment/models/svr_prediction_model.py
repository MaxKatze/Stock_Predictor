"""Preprocessed SVR Model: Support Vector Regression with technical indicator features.

Based on Konzeption Section 3.2.2 - uses 9 features:
- Historical returns: r_{t-1}, r_{t-2}, r_{t-3}
- Sentiment score: exponentially weighted average of last 10 news items
- RSI: Relative Strength Index (14 days)
- MACD: Moving Average Convergence Divergence (12/26 days)
- Bollinger Band position: normalized position within bands [0,1]
- Realized volatility: 10-day volatility of log-returns
- Trading pause indicator: binary flag for weekends/holidays (Monday effect)
"""

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
    Features follow the specification in Konzeption Section 3.2.2 (9 dimensions).
    """

    def __init__(self, C=1.0, epsilon=0.1, gamma=1.0, lookback_window=60, sentiment_data=None):
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.lookback_window = lookback_window
        self.sentiment_data = sentiment_data

        self.svr = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.data_buffer = None
        self.last_close = None

    def set_sentiment_data(self, sentiment_data):
        """Set sentiment data for the model.

        Args:
            sentiment_data: DataFrame with index=date and column 'sentiment' containing
                           sentiment scores in range [-1, 1]
        """
        self.sentiment_data = sentiment_data

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

        features_df = _compute_features(df, self.sentiment_data)
        targets = np.log(df["close"] / df["close"].shift(1)).values

        valid_mask = ~np.isnan(targets) & ~features_df.isna().any(axis=1).values
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

        features_df = _compute_features(self.data_buffer, self.sentiment_data)
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

        features_df = _compute_features(self.data_buffer, self.sentiment_data)
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
        val_features = _compute_features(val_df, self.sentiment_data)
        val_targets = np.log(val_df["close"] / val_df["close"].shift(1)).values

        valid_mask = ~np.isnan(val_targets) & ~val_features.isna().any(axis=1).values
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


def _compute_features(df: pd.DataFrame, sentiment_data: pd.DataFrame = None) -> pd.DataFrame:
    """Compute 9 features for SVR as specified in Konzeption Section 3.2.2.

    Feature vector x_t = (r_{t-1}, r_{t-2}, r_{t-3}, Sent_t, RSI_t, MACD_t, BB_t, σ_{10,t}, d_t)

    Args:
        df: DataFrame with OHLCV columns and DatetimeIndex
        sentiment_data: Optional DataFrame with sentiment scores per date

    Returns:
        DataFrame with 9 features
    """
    features = pd.DataFrame(index=df.index)

    log_returns = np.log(df["close"] / df["close"].shift(1))
    features["r_t1"] = log_returns.shift(1)
    features["r_t2"] = log_returns.shift(2)
    features["r_t3"] = log_returns.shift(3)

    if sentiment_data is not None and len(sentiment_data) > 0:
        features["sentiment"] = _compute_exponential_sentiment(df.index, sentiment_data)
    else:
        features["sentiment"] = 0.0

    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta.where(delta < 0, 0.0))
    avg_gain = gain.ewm(span=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    features["rsi"] = rsi / 100.0

    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()
    features["macd"] = ema_12 - ema_26

    sma_20 = df["close"].rolling(20).mean()
    std_20 = df["close"].rolling(20).std()
    upper_band = sma_20 + 2 * std_20
    lower_band = sma_20 - 2 * std_20
    bb_width = upper_band - lower_band
    features["bb_position"] = (df["close"] - lower_band) / (bb_width + 1e-10)
    features["bb_position"] = features["bb_position"].clip(0, 1)

    r_bar = log_returns.rolling(10).mean()
    features["volatility_10d"] = np.sqrt(
        ((log_returns - r_bar) ** 2).rolling(10).sum() / 9
    )

    features["trading_pause"] = _compute_trading_pause(df.index)

    features = features.ffill().fillna(0)
    features = features.replace([np.inf, -np.inf], 0)

    return features


def _compute_exponential_sentiment(dates: pd.DatetimeIndex, sentiment_data: pd.DataFrame) -> pd.Series:
    """Compute exponentially weighted sentiment score for each date.

    Uses the last 10 news items with exponential decay (λ=0.2).
    Formula: Sent_t = Σ(w_i * s_i) / Σ(w_i), where w_i = exp(λ * (i - 10))

    Args:
        dates: DatetimeIndex of trading days
        sentiment_data: DataFrame with 'date' index and 'sentiment' column [-1, 1]

    Returns:
        Series of sentiment scores aligned with dates
    """
    lambda_decay = 0.2
    result = pd.Series(index=dates, dtype=float)

    if sentiment_data is None or len(sentiment_data) == 0:
        return result.fillna(0)

    sentiment_data = sentiment_data.sort_index()

    for date in dates:
        past_sentiments = sentiment_data[sentiment_data.index <= date].tail(10)
        if len(past_sentiments) == 0:
            result[date] = 0.0
            continue

        n = len(past_sentiments)
        weights = np.array([np.exp(lambda_decay * (i - n + 1)) for i in range(n)])
        scores = past_sentiments["sentiment"].values if "sentiment" in past_sentiments.columns else past_sentiments.values.flatten()

        result[date] = np.sum(weights * scores) / np.sum(weights)

    return result


def _compute_trading_pause(dates: pd.DatetimeIndex) -> pd.Series:
    """Compute trading pause indicator (Monday effect).

    d_t = 1 if there was at least one non-trading day between t-1 and t, else 0.

    Args:
        dates: DatetimeIndex of trading days

    Returns:
        Series with binary indicator
    """
    result = pd.Series(index=dates, dtype=int)

    for i, date in enumerate(dates):
        if i == 0:
            result.iloc[i] = 0
            continue

        prev_date = dates[i - 1]
        days_diff = (date - prev_date).days

        result.iloc[i] = 1 if days_diff > 1 else 0

    return result
