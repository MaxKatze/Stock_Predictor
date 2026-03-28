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
    next-day prices and generate buy/sell signals.
    """

    params = dict(
        # LSTM Model Parameters
        lookback_window=60,
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
        signal_threshold=0.005, # 0.5% threshold for trading signals
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
        self.predictions = {}
        self.prices = {}
        self._next_plot_prediction = {}

        # Statistics tracking
        self.days = 0
        self.prediction_difference_sum = 0
        self.num_predictions = 0

        # Initialize per data feed
        for d in self.datas:
            self.models[d] = LSTMPredictionModel(
                lookback_window=self.p.lookback_window,
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

            self._fit_counters[d] = 0
            self._model_fitted[d] = False  # Not yet trained
            self.current_price[d] = float("nan")
            self.current_prediction[d] = None
            self.predictions[d] = []
            self.prices[d] = []
            self._next_plot_prediction[d] = None

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
            self.current_price[d] = current_price

            # Check if we have enough data
            available_bars = len(d)
            min_required = self.p.warmup_period

            if available_bars < min_required:
                self.current_prediction[d] = None
                self.predictions[d].append(None)
                self._next_plot_prediction[d] = None
                continue

            model = self.models[d]

            # Determine if we should fit/refit
            do_fit = self._should_fit(d)

            if do_fit:
                # Get OHLCV data for training
                df = self._get_ohlcv_dataframe(d)
                success = model.fit(df)

                if not success:
                    self.current_prediction[d] = None
                    self.predictions[d].append(None)
                    self._next_plot_prediction[d] = None
                    self._fit_counters[d] += 1
                    continue

                self._model_fitted[d] = True  # Mark as trained

            self._fit_counters[d] += 1

            # Append new data point if not fitting (for buffer update)
            if not do_fit and model.is_fitted:
                model.append({
                    'open': d.open[0] if hasattr(d, 'open') else d.close[0],
                    'high': d.high[0] if hasattr(d, 'high') else d.close[0],
                    'low': d.low[0] if hasattr(d, 'low') else d.close[0],
                    'close': d.close[0],
                    'volume': d.volume[0] if hasattr(d, 'volume') else 0
                })

            # Generate prediction
            prediction = model.predict(n=1)

            if pd.isna(prediction):
                self.current_prediction[d] = None
                self.predictions[d].append(None)
                self._next_plot_prediction[d] = None
                continue

            # Store prediction
            self.current_prediction[d] = prediction
            self.predictions[d].append(prediction)
            self._next_plot_prediction[d] = prediction

            # Track prediction error
            self.prediction_difference_sum += abs(prediction - current_price)
            self.num_predictions += 1

            # Generate trading signal
            pct_change = (prediction - current_price) / current_price

            position = self.getposition(d)

            if not position.size:
                # No position - check for buy signal
                if pct_change >= self.p.signal_threshold:
                    self.buy(data=d)
            else:
                # Have position - check for sell signal
                if pct_change <= -self.p.signal_threshold:
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
        else:
            print("No predictions were generated")

        print("=" * 50)
