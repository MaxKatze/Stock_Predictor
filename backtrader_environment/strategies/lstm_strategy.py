import backtrader as bt
import pandas as pd
import numpy as np

from visualization import PlotLineIndicator
from strategies import GeneralStrategy
from models.lstm_prediction_model import LSTMPredictionModel
from analyzer import (
    MeanAbsoluteErrorAnalyzer,
    RootMeanSquaredErrorAnalyzer,
    MeanSquaredErrorAnalyzer,
    RSquaredAnalyzer
)


class LSTMStrategy(GeneralStrategy):
    """
    LSTM-based trading strategy using Multi-Feature Attention LSTM.

    Uses technical indicators and attention mechanisms to predict
    n-day ahead returns and generate buy/sell signals.
    """

    params = dict(
        # LSTM Model Parameters
        lookback_window=60,
        forecast_horizon=5,  # Predict n-day ahead return
        hidden_size_1=64,
        hidden_size_2=32,
        num_heads=4,
        dropout=0.2,
        dense_units=64,
        # NOTE: bidirectional removed to avoid lookahead bias

        # Training Parameters
        epochs=50,
        batch_size=32,
        learning_rate=0.001,
        early_stopping_patience=5,

        # Strategy Parameters
        fit_interval=0,         # 0 = fit once only (no refitting to avoid lookahead)
        warmup_period=100,      # Minimum bars before first prediction
        signal_threshold=0.02,  # 2% threshold for trading signals (higher for longer horizon)

        # Pre-trained model (optional - if provided, skip training)
        pretrained_model=None,
    )

    supported_analyzers = [
        MeanAbsoluteErrorAnalyzer,
        RootMeanSquaredErrorAnalyzer,
        MeanSquaredErrorAnalyzer,
        RSquaredAnalyzer
    ]

    def __init__(self):
        super().__init__()

        # Per-data-feed state
        self.models = {}
        self.forecast_lines = {}
        self._fit_counters = {}
        self._model_fitted = {}  # Track if initial fit done
        self.current_price = {}
        self.current_prediction = {}
        self.predictions = {}  # Price predictions (for plotting)
        self.prices = {}
        self._next_plot_prediction = {}

        # Return tracking for R² calculation
        self.predicted_returns = {}  # What model predicted
        self.actual_returns = {}     # What actually happened

        # Statistics tracking
        self.days = 0
        self.prediction_difference_sum = 0
        self.num_predictions = 0

        # Initialize per data feed
        for d in self.datas:
            # Use pre-trained model if provided, otherwise create new one
            if self.p.pretrained_model is not None:
                self.models[d] = self.p.pretrained_model
                self._model_fitted[d] = True  # Already trained
            else:
                self.models[d] = LSTMPredictionModel(
                    lookback_window=self.p.lookback_window,
                    forecast_horizon=self.p.forecast_horizon,
                    hidden_size_1=self.p.hidden_size_1,
                    hidden_size_2=self.p.hidden_size_2,
                    num_heads=self.p.num_heads,
                    dropout=self.p.dropout,
                    dense_units=self.p.dense_units,
                    epochs=self.p.epochs,
                    batch_size=self.p.batch_size,
                    learning_rate=self.p.learning_rate,
                    early_stopping_patience=self.p.early_stopping_patience
                )
                self._model_fitted[d] = False  # Not yet trained

            self._fit_counters[d] = 0
            self.current_price[d] = float("nan")
            self.current_prediction[d] = None
            self.predictions[d] = []
            self.prices[d] = []
            self._next_plot_prediction[d] = None
            self.predicted_returns[d] = []
            self.actual_returns[d] = []

            # Visualization
            self.forecast_lines[d] = PlotLineIndicator()
            self.forecast_lines[d].plotinfo.plotname = f"{d._name} LSTM"

    def _get_ohlcv_dataframe(self, d):
        """Extract OHLCV data from backtrader data feed."""
        size = len(d)

        # Get available data
        closes = list(d.close.get(size=size))
        opens = list(d.open.get(size=size)) if hasattr(d, 'open') else closes
        highs = list(d.high.get(size=size)) if hasattr(d, 'high') else closes
        lows = list(d.low.get(size=size)) if hasattr(d, 'low') else closes
        volumes = list(d.volume.get(size=size)) if hasattr(d, 'volume') else [0] * len(closes)

        df = pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        })

        return df

    def _should_fit(self, d):
        """Determine if model needs (re)fitting."""
        # If never fitted and we have enough data, fit now
        if not self._model_fitted[d]:
            return True

        # If fit_interval=0, never refit after initial training
        if self.p.fit_interval == 0:
            return False

        # Otherwise refit periodically
        return self._fit_counters[d] % self.p.fit_interval == 0

    def next(self):
        self.days += 1

        for d in self.datas:
            # Update visualization with delayed prediction
            delayed_prediction = self._next_plot_prediction[d]
            self.forecast_lines[d].lines.line[0] = (
                float("nan") if delayed_prediction is None else delayed_prediction
            )

            # Record current price
            current_price = d.close[0]
            self.prices[d].append(current_price)

            # Calculate actual n-day return (comparing to forecast_horizon days ago)
            # This is the return we were trying to predict forecast_horizon days ago
            horizon = self.p.forecast_horizon
            if len(self.prices[d]) > horizon:
                past_price = self.prices[d][-(horizon + 1)]
                actual_return = (current_price - past_price) / past_price
                self.actual_returns[d].append(actual_return)

            self.current_price[d] = current_price

            model = self.models[d]

            # Check if we have enough data
            available_bars = len(d)

            # If model is pre-trained, we only need lookback_window bars
            # Otherwise we need warmup_period for training
            if self._model_fitted[d]:
                min_required = model.lookback_window
            else:
                min_required = self.p.warmup_period

            if available_bars < min_required:
                self.current_prediction[d] = None
                self.predictions[d].append(None)
                self.predicted_returns[d].append(None)
                self._next_plot_prediction[d] = None
                continue

            # Determine if we should fit/refit
            do_fit = self._should_fit(d)

            if do_fit:
                # Get OHLCV data for training
                df = self._get_ohlcv_dataframe(d)
                success = model.fit(df)

                if not success:
                    self.current_prediction[d] = None
                    self.predictions[d].append(None)
                    self.predicted_returns[d].append(None)
                    self._next_plot_prediction[d] = None
                    self._fit_counters[d] += 1
                    continue

                self._model_fitted[d] = True  # Mark as trained

            self._fit_counters[d] += 1

            # Update model's data buffer with current window from test data
            # This is needed for pre-trained models that don't have test data in buffer
            if model.is_fitted:
                # Get current OHLCV window and set as buffer
                df = self._get_ohlcv_dataframe(d)
                df.columns = [c.lower() for c in df.columns]
                model.data_buffer = df.copy()
                model.last_close = df['close'].iloc[-1]

            # Generate prediction (price prediction)
            prediction = model.predict(n=1)

            if pd.isna(prediction):
                self.current_prediction[d] = None
                self.predictions[d].append(None)
                self.predicted_returns[d].append(None)
                self._next_plot_prediction[d] = None
                continue

            # Store price prediction (for plotting and MAE/MSE)
            self.current_prediction[d] = prediction
            self.predictions[d].append(prediction)
            self._next_plot_prediction[d] = prediction

            # Store predicted return (for R² calculation)
            predicted_return = (prediction - current_price) / current_price
            self.predicted_returns[d].append(predicted_return)

            # Track prediction error
            self.prediction_difference_sum += abs(prediction - current_price)
            self.num_predictions += 1

            # Generate trading signal based on predicted return
            position = self.getposition(d)

            if not position.size:
                # No position - check for buy signal
                if predicted_return >= self.p.signal_threshold:
                    self.buy(data=d)
            else:
                # Have position - check for sell signal
                if predicted_return <= -self.p.signal_threshold:
                    self.close(data=d)

    def stop(self):
        """Called when backtest ends."""
        print("\n" + "=" * 50)
        print("LSTM Strategy Results")
        print("=" * 50)

        if self.num_predictions > 0:
            avg_diff = self.prediction_difference_sum / self.num_predictions
            print(f"Total trading days: {self.days}")
            print(f"Predictions made: {self.num_predictions}")
            print(f"Average prediction error: ${avg_diff:.4f}")

            # Debug: Show predicted return statistics
            for d in self.datas:
                valid_returns = [r for r in self.predicted_returns[d] if r is not None]
                if valid_returns:
                    import numpy as np
                    print(f"  [{d._name}] Predicted returns: min={min(valid_returns)*100:.2f}%, "
                          f"max={max(valid_returns)*100:.2f}%, mean={np.mean(valid_returns)*100:.2f}%")
        else:
            print("No predictions were generated")

        print("=" * 50)
