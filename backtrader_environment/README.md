# Stock Predictor (Backtrader + ARIMA)

Python project for running Backtrader strategies with custom prediction models and analyzers.

## Requirements

- Python `>=3.13`
- `uv` (recommended)

## Run

```bash
uv run main.py
```

## Project Structure

- `main.py`: Entry point, registers strategy, analyzers, sizer, and data feeds.
- `strategies/`: Trading logic (`ARIMAStrategy`, moving average, fixed date).
- `models/`: Prediction model implementations (currently ARIMA).
- `analyzer/`: Custom prediction error metrics.
- `sizing/`: Position sizing logic.
- `data_handling/`: Download and format market data.
- `visualization/`: Custom Backtrader indicator lines for plotting.

## Flow

1. A strategy is added in `main.py`.
2. The strategy builds/updates a prediction model.
3. The strategy generates trade decisions from model output.
4. Analyzers evaluate prediction quality and portfolio metrics.
