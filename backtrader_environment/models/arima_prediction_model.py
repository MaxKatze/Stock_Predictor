from .prediction_models import PredictionModel
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
import numpy as np


class ARIMAModel(PredictionModel):
    def __init__(self, order=(1, 1, 1), window=100):
        self.order = order
        self.window = window
        self.model_fit: ARIMAResults | None = None

    def append(self, data) -> bool:
        if self.model_fit is None:
            return False

        try:
            self.model_fit = self.model_fit.append(data, refit=False)
            return True
        except Exception as exc:
            print(f"ARIMA append failed: {exc}")
            self.model_fit = None
            return False

    def predict(self, n=1):
        if self.model_fit is None:
            return np.nan

        forecast = self.model_fit.forecast(steps=n)
        return forecast.iloc[-1] if n == 1 else forecast.values
    
    def fit(self, data, window=None):
        fit_window = self.window if window is None else window
        if not isinstance(data, pd.Series):
            data = pd.Series(data)
        
        if len(data) > fit_window:
            data = data.iloc[-fit_window:]

        try:
            model = ARIMA(data, order=self.order)
            self.model_fit = model.fit()
            return True
        except Exception as exc:
            print(f"ARIMA fit failed: {exc}")
            self.model_fit = None
            return False
