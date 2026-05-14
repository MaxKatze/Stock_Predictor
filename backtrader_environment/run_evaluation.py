"""Walk-Forward Evaluation Framework.

Runs all prediction models on selected stocks with proper 60/10/30 data split.
Produces comparison tables with prediction metrics and financial performance.
"""

import os
import sys
import pandas as pd
import numpy as np
import backtrader as bt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_handling.data_loader import load_config, get_data_split, get_selected_stocks
from data_handling.file_format_yahoo import PandasData
from models.linear_separation_model import LinearSeparationModel
from models.svr_prediction_model import SVRPredictionModel
from models.arima_prediction_model import ARIMAModel
from strategies.prediction_strategy import PredictionStrategy
from sizing.PercentageSizing import PercentageSizer
from analyzer import (
    MeanAbsoluteErrorAnalyzer,
    RootMeanSquaredErrorAnalyzer,
    MAPEAnalyzer,
    OOSRSquaredAnalyzer,
    DirectionalAccuracyAnalyzer,
)

try:
    from models.timesfm_prediction_model import TimesFMPredictionModel, _TIMESFM_AVAILABLE
except ImportError:
    _TIMESFM_AVAILABLE = False


INITIAL_CASH = 100_000.0


def train_linear_separation(train_df, val_df):
    """Train LinearSeparation model."""
    model = LinearSeparationModel()
    model.fit(train_df["Close"].values)
    return model


def train_svr(train_df, val_df):
    """Train SVR model with grid search on validation data."""
    model = SVRPredictionModel()
    model.fit(train_df, validation_data=val_df)
    return model


def train_arima(train_df, val_df):
    """Train ARIMA model."""
    model = ARIMAModel(order=(2, 1, 1), window=100)
    model.fit(train_df["Close"].values, window=100)
    return model


def train_timesfm(train_df, val_df):
    """Initialize TimesFM (zero-shot, no real training)."""
    if not _TIMESFM_AVAILABLE:
        return None
    model = TimesFMPredictionModel()
    model.fit(train_df)
    return model


MODEL_CONFIGS = {
    "LinearSeparation": {
        "train_fn": train_linear_separation,
        "signal_threshold": 0.005,
        "forecast_horizon": 1,
        "warmup_period": 60,
    },
    "SVR": {
        "train_fn": train_svr,
        "signal_threshold": 0.005,
        "forecast_horizon": 1,
        "warmup_period": 60,
    },
    # "ARIMA": {
    #     "train_fn": train_arima,
    #     "signal_threshold": 0.001,
    #     "forecast_horizon": 1,
    #     "warmup_period": 100,
    #     "fit_interval": 100,
    # },
}

if _TIMESFM_AVAILABLE:
    MODEL_CONFIGS["TimesFM"] = {
        "train_fn": train_timesfm,
        "signal_threshold": 0.005,
        "forecast_horizon": 1,
        "warmup_period": 60,
    }


def run_backtest(model, test_df, model_config):
    """Run a backtrader backtest with the given model on test data."""
    cerebro = bt.Cerebro()

    data = PandasData(dataname=test_df)
    cerebro.adddata(data)

    cerebro.addstrategy(
        PredictionStrategy,
        model=model,
        signal_threshold=model_config.get("signal_threshold", 0.01),
        forecast_horizon=model_config.get("forecast_horizon", 1),
        warmup_period=model_config.get("warmup_period", 60),
        fit_interval=model_config.get("fit_interval", 0),
    )

    cerebro.addsizer(PercentageSizer)
    cerebro.broker.setcash(INITIAL_CASH)

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=0.02)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(MeanAbsoluteErrorAnalyzer, _name="mae")
    cerebro.addanalyzer(RootMeanSquaredErrorAnalyzer, _name="rmse")
    cerebro.addanalyzer(MAPEAnalyzer, _name="mape")
    cerebro.addanalyzer(OOSRSquaredAnalyzer, _name="oos_r2")
    cerebro.addanalyzer(DirectionalAccuracyAnalyzer, _name="da")

    results = cerebro.run()
    strat = results[0]

    final_value = cerebro.broker.getvalue()
    total_return = (final_value - INITIAL_CASH) / INITIAL_CASH

    bh_return = (test_df["Close"].iloc[-1] - test_df["Close"].iloc[0]) / test_df["Close"].iloc[0]

    sharpe_analysis = strat.analyzers.sharpe.get_analysis()
    sharpe = sharpe_analysis.get("sharperatio", None)

    dd_analysis = strat.analyzers.drawdown.get_analysis()
    max_dd = dd_analysis.get("max", {}).get("drawdown", 0) / 100

    mae_analysis = strat.analyzers.mae.get_analysis()
    rmse_analysis = strat.analyzers.rmse.get_analysis()
    mape_analysis = strat.analyzers.mape.get_analysis()
    oos_r2_analysis = strat.analyzers.oos_r2.get_analysis()
    da_analysis = strat.analyzers.da.get_analysis()

    data_name = list(mae_analysis.get("mean_absolute_error", {}).keys())[0] if mae_analysis.get("mean_absolute_error") else ""

    return {
        "total_return": total_return,
        "bh_return": bh_return,
        "excess_return": total_return - bh_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "mae": mae_analysis.get("mean_absolute_error", {}).get(data_name, -1),
        "rmse": rmse_analysis.get("root_mean_squared_error", {}).get(data_name, -1),
        "mape": mape_analysis.get("mape", {}).get(data_name, -1),
        "oos_r2": oos_r2_analysis.get("oos_r_squared", {}).get(data_name, -1),
        "directional_accuracy": da_analysis.get("directional_accuracy", {}).get(data_name, -1),
    }


def main():
    config = load_config()
    stocks = get_selected_stocks(config)

    print("=" * 80)
    print("WALK-FORWARD MODEL EVALUATION")
    print("=" * 80)
    print(f"Stocks: {stocks}")
    print(f"Models: {list(MODEL_CONFIGS.keys())}")
    print(f"Initial Cash: ${INITIAL_CASH:,.0f}")
    print()

    all_results = []

    for ticker in stocks:
        print(f"\n{'─' * 60}")
        print(f"Evaluating: {ticker}")
        print(f"{'─' * 60}")

        try:
            train_df, val_df, test_df = get_data_split(ticker, config)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        if len(test_df) < 60:
            print(f"  SKIP: Not enough test data ({len(test_df)} rows)")
            continue

        print(f"  Train: {len(train_df)} days, Val: {len(val_df)} days, Test: {len(test_df)} days")

        for model_name, model_config in MODEL_CONFIGS.items():
            print(f"  Running {model_name}...", end=" ")

            try:
                model = model_config["train_fn"](train_df, val_df)
                if model is None:
                    print("SKIPPED (model not available)")
                    continue

                metrics = run_backtest(model, test_df, model_config)
                metrics["ticker"] = ticker
                metrics["model"] = model_name
                all_results.append(metrics)

                print(f"Return: {metrics['total_return']:.2%}, "
                      f"DA: {metrics['directional_accuracy']:.3f}, "
                      f"OOS-R²: {metrics['oos_r2']:.4f}")

            except Exception as e:
                print(f"ERROR: {e}")
                continue

    if all_results:
        print("\n\n")
        print("=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)

        df = pd.DataFrame(all_results)
        cols = ["ticker", "model", "total_return", "bh_return", "excess_return",
                "sharpe_ratio", "max_drawdown", "mape", "oos_r2", "directional_accuracy"]
        print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        print("\n\nAverage by Model:")
        numeric_cols = ["total_return", "excess_return", "sharpe_ratio",
                        "max_drawdown", "mape", "oos_r2", "directional_accuracy"]
        print(df.groupby("model")[numeric_cols].mean().to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
