
from .prediction_models import PredictionModel
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

class ARIMAModel(PredictionModel):
    def __init__(self, order=(1,1,1), window=100):
        """
        order: tuple - (p,d,q) Parameter für ARIMA
        window: int - wie viele letzte Datenpunkte zur Schätzung verwendet werden
        """
        self.order = order
        self.window = window
        self.model_fit = None

    def predict(self, n=1):
        """
        data: pandas.Series oder list - Zeitreihe (z. B. Schlusskurse)
        n: int - wie viele Schritte in die Zukunft prognostiziert werden sollen
        returns: float oder np.ndarray - Prognosewert(e)
        """       
        # Prognose
        forecast = self.model_fit.forecast(steps=n)

        return forecast.iloc[-1] if n == 1 else forecast.values
    

    def fit(self, data, window):
        if not isinstance(data, pd.Series):
            data = pd.Series(data)
        
        # aktuell bis window
        if len(data) > window:
            data = data.iloc[-window:]

        # fit
        try:
            model = ARIMA(data, order=self.order)
            self.model_fit = model.fit()
        except Exception as e:
            print(f"ARIMA fit failed: {e}")
            return np.nan