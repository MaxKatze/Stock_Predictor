import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from models.prediction_models import PredictionModel


class FeatureAttention(nn.Module):
    """Learnable attention weights for input features."""

    def __init__(self, num_features):
        super().__init__()
        self.attention_weights = nn.Parameter(torch.ones(num_features))

    def forward(self, x):
        # x: (batch, seq_len, features)
        weights = torch.softmax(self.attention_weights, dim=0)
        return x * weights.unsqueeze(0).unsqueeze(0)


class TemporalAttention(nn.Module):
    """Multi-head attention over temporal sequence."""

    def __init__(self, hidden_size, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
        attn_out, _ = self.attention(x, x, x)
        return self.layer_norm(x + attn_out)


class MFALSTMNetwork(nn.Module):
    """
    Multi-Feature Attention LSTM Network for return prediction.

    NOTE: Uses unidirectional LSTM to avoid lookahead bias.
    Bidirectional LSTM would look at future data in the sequence.
    """

    def __init__(
        self,
        num_features=14,
        hidden_size_1=64,
        hidden_size_2=32,
        num_heads=4,
        dropout=0.2,
        dense_units=64,
    ):
        super().__init__()

        # Feature attention
        self.feature_attention = FeatureAttention(num_features)

        # First LSTM layer - UNIDIRECTIONAL to avoid lookahead bias
        self.lstm1 = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size_1,
            batch_first=True,
            bidirectional=False  # IMPORTANT: No bidirectional!
        )

        self.dropout1 = nn.Dropout(dropout)

        # Second LSTM layer - UNIDIRECTIONAL
        self.lstm2 = nn.LSTM(
            input_size=hidden_size_1,
            hidden_size=hidden_size_2,
            batch_first=True,
            bidirectional=False  # IMPORTANT: No bidirectional!
        )

        # Temporal attention (only on past, not future)
        self.temporal_attention = TemporalAttention(hidden_size_2, num_heads)

        # Dense layers - output is a single return value
        self.dense = nn.Sequential(
            nn.Linear(hidden_size_2, dense_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_units, 1)
        )

    def forward(self, x):
        # x: (batch, seq_len, features)

        # Apply feature attention
        x = self.feature_attention(x)

        # First LSTM
        x, _ = self.lstm1(x)
        x = self.dropout1(x)

        # Second LSTM
        x, _ = self.lstm2(x)

        # Temporal attention
        x = self.temporal_attention(x)

        # Use last timestep for prediction (only uses past information)
        x = x[:, -1, :]

        # Dense layers
        output = self.dense(x)

        return output.squeeze(-1)


class LSTMPredictionModel(PredictionModel):
    """
    Multi-Feature Attention LSTM model for stock RETURN prediction.

    IMPORTANT: This implementation avoids lookahead bias by:
    1. Using unidirectional LSTM (not bidirectional)
    2. Fitting scaler only on training data
    3. Using only forward-fill for NaN values (not backward-fill)
    4. Properly shifting targets for n-day ahead prediction
    """

    def __init__(
        self,
        lookback_window=60,
        forecast_horizon=5,  # Predict n-day ahead return (default: 5 days)
        hidden_size_1=64,
        hidden_size_2=32,
        num_heads=4,
        dropout=0.2,
        dense_units=64,
        epochs=50,
        batch_size=32,
        learning_rate=0.001,
        early_stopping_patience=5,
        device=None
    ):
        self.lookback_window = lookback_window
        self.forecast_horizon = forecast_horizon
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.num_heads = num_heads
        self.dropout = dropout
        self.dense_units = dense_units
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.early_stopping_patience = early_stopping_patience

        # Auto-detect device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = None
        self.feature_scaler = StandardScaler()
        self.is_fitted = False

        # Data buffer for predictions
        self.data_buffer = None
        self.last_close = None
        self.num_features = 14

    def _compute_features(self, df):
        """
        Compute technical indicators from OHLCV data.

        IMPORTANT: Only uses past data, no lookahead bias.
        - All rolling windows look backwards
        - Only forward-fill for NaN (no backward-fill)
        """
        features = pd.DataFrame(index=df.index)

        # Return-based price features (all look backwards)
        features['return_1d'] = df['close'].pct_change(1)
        features['return_5d'] = df['close'].pct_change(5)
        features['return_10d'] = df['close'].pct_change(10)
        features['return_20d'] = df['close'].pct_change(20)

        # High-low range as percentage
        if 'high' in df.columns and 'low' in df.columns:
            features['hl_range'] = (df['high'] - df['low']) / df['close']
        else:
            features['hl_range'] = 0

        # Open-close change
        if 'open' in df.columns:
            features['oc_change'] = (df['close'] - df['open']) / df['open']
        else:
            features['oc_change'] = 0

        # Volume change (log-scaled)
        if 'volume' in df.columns and df['volume'].sum() > 0:
            features['volume_change'] = np.log1p(df['volume']).pct_change()
        else:
            features['volume_change'] = 0

        # Volatility (rolling std of returns - looks backwards)
        features['volatility_5d'] = df['close'].pct_change().rolling(5).std()
        features['volatility_20d'] = df['close'].pct_change().rolling(20).std()

        # RSI (rolling calculation - looks backwards)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        features['rsi_14'] = (100 - (100 / (1 + rs))) / 100

        # MACD (EMA looks backwards)
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        features['macd_pct'] = macd / df['close']
        features['macd_signal_pct'] = macd.ewm(span=9, adjust=False).mean() / df['close']

        # SMA ratios (rolling mean looks backwards)
        sma_20 = df['close'].rolling(20).mean()
        sma_50 = df['close'].rolling(50).mean()
        features['sma_ratio_20'] = df['close'] / (sma_20 + 1e-10) - 1
        features['sma_ratio_50'] = df['close'] / (sma_50 + 1e-10) - 1

        # ONLY forward-fill NaN values (NO backward-fill to avoid lookahead!)
        features = features.ffill()
        # Fill remaining NaN at the start with 0
        features = features.fillna(0)

        # Replace infinities
        features = features.replace([np.inf, -np.inf], 0)

        return features

    def _prepare_sequences(self, features, targets=None):
        """
        Create sequences for LSTM input.

        For each sequence at position i:
        - X: Features from [i - lookback_window + 1, i + 1) = days ending at i (inclusive)
        - y: Target at i = return from day i to day i+1

        This means we use features UP TO AND INCLUDING day i to predict
        the return from day i to day i+1. No lookahead bias.
        """
        X = []
        y = [] if targets is not None else None

        # Start from lookback_window - 1 so we can include index i in the sequence
        for i in range(self.lookback_window - 1, len(features)):
            # Sequence includes features from (i - lookback_window + 1) to i (inclusive)
            # That's lookback_window elements: [i-59, i-58, ..., i-1, i] for lookback=60
            X.append(features[i - self.lookback_window + 1:i + 1])
            if targets is not None:
                y.append(targets[i])

        X = np.array(X)
        if y is not None:
            y = np.array(y)
            return X, y
        return X

    def fit(self, data, window=None):
        """
        Fit the LSTM model on historical data to predict RETURNS.

        IMPORTANT: Scaler is fitted only on training data, not validation data.
        """
        try:
            # Convert to DataFrame if Series
            if isinstance(data, pd.Series):
                df = pd.DataFrame({'close': data})
            else:
                df = data.copy()

            # Ensure column names are lowercase
            df.columns = [c.lower() for c in df.columns]

            # Store last close for prediction conversion
            self.last_close = df['close'].iloc[-1]

            # Store in buffer
            self.data_buffer = df.copy()

            # Need enough data for features + sequences
            min_data_needed = self.lookback_window + 60
            if len(df) < min_data_needed:
                return False

            # Compute features
            features_df = self._compute_features(df)

            # Target is N-DAY AHEAD RETURN (shifted by -forecast_horizon)
            # e.g., forecast_horizon=5: predict return from day i to day i+5
            future_returns = df['close'].pct_change(self.forecast_horizon).shift(-self.forecast_horizon)

            # Remove rows without valid target
            valid_idx = ~future_returns.isna()
            features_df = features_df[valid_idx]
            future_returns = future_returns[valid_idx]

            # Get feature values and targets
            feature_values = features_df.values
            targets = future_returns.values

            # Clip extreme returns (adjusted for longer horizon)
            max_return = 0.2 * (self.forecast_horizon / 5)  # Scale with horizon
            targets = np.clip(targets, -max_return, max_return)

            # Train/validation split BEFORE scaling (to avoid lookahead)
            split_idx = int(len(feature_values) * 0.8)

            # Fit scaler ONLY on training data
            train_features = feature_values[:split_idx]
            val_features = feature_values[split_idx:]

            self.feature_scaler.fit(train_features)

            # Transform both sets using scaler fitted on training only
            train_features_scaled = self.feature_scaler.transform(train_features)
            val_features_scaled = self.feature_scaler.transform(val_features)

            train_targets = targets[:split_idx]
            val_targets = targets[split_idx:]

            # Create sequences
            X_train, y_train = self._prepare_sequences(train_features_scaled, train_targets)
            X_val, y_val = self._prepare_sequences(val_features_scaled, val_targets)

            if len(X_train) < self.batch_size:
                return False

            # Convert to tensors
            X_train = torch.FloatTensor(X_train).to(self.device)
            y_train = torch.FloatTensor(y_train).to(self.device)
            X_val = torch.FloatTensor(X_val).to(self.device)
            y_val = torch.FloatTensor(y_val).to(self.device)

            # Initialize model (unidirectional LSTM)
            self.model = MFALSTMNetwork(
                num_features=self.num_features,
                hidden_size_1=self.hidden_size_1,
                hidden_size_2=self.hidden_size_2,
                num_heads=self.num_heads,
                dropout=self.dropout,
                dense_units=self.dense_units,
            ).to(self.device)

            # Loss and optimizer
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

            # Training loop with early stopping
            best_val_loss = float('inf')
            patience_counter = 0
            best_model_state = None

            for epoch in range(self.epochs):
                self.model.train()

                # Mini-batch training
                indices = torch.randperm(len(X_train))
                for i in range(0, len(X_train), self.batch_size):
                    batch_indices = indices[i:i + self.batch_size]
                    X_batch = X_train[batch_indices]
                    y_batch = y_train[batch_indices]

                    optimizer.zero_grad()
                    predictions = self.model(X_batch)
                    loss = criterion(predictions, y_batch)
                    loss.backward()
                    optimizer.step()

                # Validation
                self.model.eval()
                with torch.no_grad():
                    if len(X_val) > 0:
                        val_predictions = self.model(X_val)
                        val_loss = criterion(val_predictions, y_val).item()
                    else:
                        val_loss = best_val_loss  # No validation data

                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        break

            # Restore best model
            if best_model_state is not None:
                self.model.load_state_dict(best_model_state)

            self.is_fitted = True
            return True

        except Exception as e:
            print(f"LSTM fit error: {e}")
            import traceback
            traceback.print_exc()
            self.is_fitted = False
            return False

    def predict(self, n=1):
        """
        Predict next n PRICES by predicting returns and converting.
        """
        if not self.is_fitted or self.model is None or self.data_buffer is None:
            return np.nan

        try:
            self.model.eval()

            # Compute features for current data
            features_df = self._compute_features(self.data_buffer)
            feature_values = features_df.values

            # Scale features using the scaler fitted during training
            feature_values_scaled = self.feature_scaler.transform(feature_values)

            # Get last lookback_window
            if len(feature_values_scaled) < self.lookback_window:
                return np.nan

            last_sequence = feature_values_scaled[-self.lookback_window:]

            # Get current price for conversion
            current_price = self.data_buffer['close'].iloc[-1]

            predictions = []
            current_sequence = last_sequence.copy()
            price = current_price

            for _ in range(n):
                # Predict return
                X = torch.FloatTensor(current_sequence).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    predicted_return = self.model(X).cpu().numpy()[0]

                # Convert return to price
                price = price * (1 + predicted_return)
                predictions.append(price)

                # For multi-step, shift sequence
                if n > 1:
                    current_sequence = np.vstack([current_sequence[1:], current_sequence[-1]])

            if n == 1:
                return predictions[0]
            return np.array(predictions)

        except Exception as e:
            print(f"LSTM predict error: {e}")
            return np.nan

    def predict_return(self):
        """Predict n-day ahead return directly (where n = forecast_horizon)."""
        if not self.is_fitted or self.model is None or self.data_buffer is None:
            return np.nan

        try:
            self.model.eval()

            features_df = self._compute_features(self.data_buffer)
            feature_values = features_df.values
            feature_values_scaled = self.feature_scaler.transform(feature_values)

            if len(feature_values_scaled) < self.lookback_window:
                return np.nan

            last_sequence = feature_values_scaled[-self.lookback_window:]
            X = torch.FloatTensor(last_sequence).unsqueeze(0).to(self.device)

            with torch.no_grad():
                predicted_return = self.model(X).cpu().numpy()[0]

            return float(predicted_return)

        except Exception as e:
            print(f"LSTM predict_return error: {e}")
            return np.nan

    def append(self, new_data):
        """Append new data point to buffer."""
        if self.data_buffer is None:
            return False

        try:
            if isinstance(new_data, dict):
                new_row = pd.DataFrame([new_data])
            elif isinstance(new_data, (int, float)):
                new_row = pd.DataFrame([{'close': new_data}])
            elif isinstance(new_data, pd.DataFrame):
                new_row = new_data
            else:
                return False

            new_row.columns = [c.lower() for c in new_row.columns]
            self.data_buffer = pd.concat([self.data_buffer, new_row], ignore_index=True)

            self.last_close = self.data_buffer['close'].iloc[-1]

            # Keep buffer manageable
            max_buffer_size = self.lookback_window * 3
            if len(self.data_buffer) > max_buffer_size:
                self.data_buffer = self.data_buffer.iloc[-max_buffer_size:]

            return True

        except Exception as e:
            print(f"LSTM append error: {e}")
            return False
