"""TimesFM Model: Google TimesFM Foundation Model for zero-shot time series forecasting."""

import numpy as np
import pandas as pd
import warnings

from .prediction_models import PredictionModel

_TIMESFM_AVAILABLE = False
_TIMESFM_SHARED_MODEL = None  # Shared model instance across all instances

try:
    import timesfm
    _TIMESFM_AVAILABLE = True
except ImportError:
    pass


def _get_shared_timesfm_model(backend="cpu", horizon=1):
    """Get or create shared TimesFM model instance.

    This ensures the model is only downloaded and loaded once,
    regardless of how many TimesFMPredictionModel instances are created.
    """
    global _TIMESFM_SHARED_MODEL

    if _TIMESFM_SHARED_MODEL is None and _TIMESFM_AVAILABLE:
        try:
            _TIMESFM_SHARED_MODEL = timesfm.TimesFm(
                hparams=timesfm.TimesFmHparams(
                    backend=backend,
                    per_core_batch_size=32,
                    horizon_len=horizon,
                ),
                checkpoint=timesfm.TimesFmCheckpoint(
                    huggingface_repo_id="google/timesfm-1.0-200m-pytorch"
                ),
            )
        except Exception as e:
            warnings.warn(f"Failed to initialize TimesFM: {e}")
            _TIMESFM_SHARED_MODEL = None

    return _TIMESFM_SHARED_MODEL


class TimesFMPredictionModel(PredictionModel):
    """TimesFM zero-shot prediction model for stock returns.

    Uses Google's pre-trained TimesFM foundation model without fine-tuning.
    Operates on log-returns for consistency with other models.

    Requires: pip install timesfm (optional dependency)
    """

    def __init__(self, context_length=512, horizon=1, backend="cpu"):
        self.context_length = context_length
        self.horizon = horizon
        self.backend = backend

        self.tfm = None
        self.data_buffer = None
        self.last_close = None
        self.is_fitted = False

        if _TIMESFM_AVAILABLE:
            self.tfm = _get_shared_timesfm_model(backend=backend, horizon=horizon)

    def fit(self, data, window=None):
        """Store data buffer (no training for zero-shot model).

        Args:
            data: DataFrame or Series of close prices
            window: ignored
        """
        if not _TIMESFM_AVAILABLE:
            warnings.warn("timesfm package not installed. Install with: pip install timesfm")
            self.is_fitted = False
            return

        if isinstance(data, pd.Series):
            df = pd.DataFrame({"close": data})
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
            df.columns = [c.lower() for c in df.columns]
        else:
            df = pd.DataFrame({"close": np.array(data, dtype=float)})

        self.data_buffer = df
        self.last_close = df["close"].iloc[-1]
        self.is_fitted = self.tfm is not None

    def predict(self, n=1):
        """Predict next price(s) using TimesFM on log-returns."""
        if not self.is_fitted or self.data_buffer is None:
            return float("nan") if n == 1 else np.full(n, float("nan"))

        closes = self.data_buffer["close"].values
        log_returns = np.diff(np.log(closes))

        context = log_returns[-self.context_length:]

        try:
            forecast_input = [context.tolist()]
            point_forecast, _ = self.tfm.forecast(
                forecast_input,
                freq=[0],
            )
            predicted_returns = point_forecast[0][:n]
        except Exception as e:
            warnings.warn(f"TimesFM forecast failed: {e}")
            return float("nan") if n == 1 else np.full(n, float("nan"))

        current_price = self.last_close
        predictions = []
        for r in predicted_returns:
            next_price = current_price * np.exp(r)
            predictions.append(next_price)
            current_price = next_price

        if n == 1:
            return predictions[0]
        return np.array(predictions)

    def predict_return(self):
        """Predict next log-return directly."""
        if not self.is_fitted or self.data_buffer is None:
            return float("nan")

        closes = self.data_buffer["close"].values
        log_returns = np.diff(np.log(closes))
        context = log_returns[-self.context_length:]

        try:
            forecast_input = [context.tolist()]
            point_forecast, _ = self.tfm.forecast(forecast_input, freq=[0])
            return point_forecast[0][0]
        except Exception:
            return float("nan")

    def update(self, new_price):
        """Update model with new price observation."""
        if self.data_buffer is not None:
            new_df = pd.DataFrame({"close": [float(new_price)]})
            self.data_buffer = pd.concat([self.data_buffer, new_df], ignore_index=True)
            self.last_close = float(new_price)

    def append(self, new_data):
        """Append new data to buffer for walk-forward."""
        if self.data_buffer is not None:
            if isinstance(new_data, pd.DataFrame):
                new_df = new_data.copy()
                new_df.columns = [c.lower() for c in new_df.columns]
            elif isinstance(new_data, pd.Series):
                new_df = pd.DataFrame({"close": [new_data.iloc[-1]]})
            else:
                new_df = pd.DataFrame({"close": [float(new_data[-1])]})
            self.data_buffer = pd.concat([self.data_buffer, new_df], ignore_index=True)
            self.last_close = self.data_buffer["close"].iloc[-1]
