"""Walk-Forward Evaluation Framework.

Runs all prediction models on selected stocks with proper 60/10/30 data split.
Produces comparison tables with prediction metrics and financial performance.
Saves results to JSON and generates example walk-forward plots.

Usage:
    uv run python run_evaluation.py                     # Run all available models
    uv run python run_evaluation.py --models LinearSeparation SVR
    uv run python run_evaluation.py --models TimesFM
    uv run python run_evaluation.py --list-models       # Show available models
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import backtrader as bt
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_handling.data_loader import load_config, get_data_split, get_selected_stocks
from data_handling.file_format_yahoo import PandasData
from data_handling.sentiment_loader import (
    get_sentiment_for_model,
    generate_placeholder_sentiment,
)
from models.linear_separation_model import LinearSeparationModel
from models.svr_prediction_model import SVRPredictionModel
from models.arima_prediction_model import ARIMAModel
from strategies.prediction_strategy import PredictionStrategy
from sizing.HalfKellySizer import HalfKellyPositionManager
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
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT_DIR, "data", "evaluation")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def compute_validation_mae(model, val_df):
    """Compute MAE on validation data to determine signal threshold ε."""
    if model is None or len(val_df) < 60:
        return 0.005

    closes = val_df["Close"].values
    errors = []

    df_lower = val_df.copy()
    df_lower.columns = [c.lower() for c in df_lower.columns]

    if hasattr(model, "data_buffer"):
        model.data_buffer = df_lower
        model.last_close = closes[-1]

    for i in range(60, len(closes) - 1):
        if hasattr(model, "data_buffer"):
            model.data_buffer = df_lower.iloc[: i + 1]
            model.last_close = closes[i]

        try:
            prediction = model.predict(n=1)
            if prediction is None or np.isnan(prediction):
                continue

            actual_return = np.log(closes[i + 1] / closes[i])
            predicted_return = np.log(prediction / closes[i])
            errors.append(abs(predicted_return - actual_return))
        except Exception:
            continue

    if len(errors) == 0:
        return 0.005

    return np.mean(errors)


def train_linear_separation(train_df, val_df):
    """Train LinearSeparation model."""
    model = LinearSeparationModel()
    model.fit_order(train_df["Close"].values)
    model.fit(train_df["Close"].values)
    return model


def train_svr(train_df, val_df, sentiment_data=None):
    """Train SVR model with grid search on validation data."""
    model = SVRPredictionModel()
    if sentiment_data is not None:
        model.set_sentiment_data(sentiment_data)
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
        "forecast_horizon": 1,
        "warmup_period": 60,
    },
    "SVR": {
        "train_fn": train_svr,
        "forecast_horizon": 1,
        "warmup_period": 60,
    },
}

if _TIMESFM_AVAILABLE:
    MODEL_CONFIGS["TimesFM"] = {
        "train_fn": train_timesfm,
        "forecast_horizon": 1,
        "warmup_period": 60,
    }
else:
    print("INFO: TimesFM nicht verfügbar (nicht installiert). Überspringe TimesFM-Modell.")


class WalkForwardObserver(bt.Observer):
    """Observer to track predictions and signals for plotting."""
    lines = ('prediction', 'signal',)
    plotinfo = dict(plot=True, subplot=True)

    def next(self):
        strat = self._owner
        if hasattr(strat, 'current_prediction') and strat.datas[0] in strat.current_prediction:
            pred = strat.current_prediction[strat.datas[0]]
            self.lines.prediction[0] = pred if pred is not None else float('nan')
        else:
            self.lines.prediction[0] = float('nan')


def run_backtest(model, test_df, model_config, generate_plot=False, plot_path=None, ticker="", model_name=""):
    """Run a backtrader backtest with the given model on test data."""
    cerebro = bt.Cerebro()

    data = PandasData(dataname=test_df)
    cerebro.adddata(data, name=ticker)

    cerebro.addstrategy(
        PredictionStrategy,
        model=model,
        signal_threshold=model_config.get("signal_threshold", 0.01),
        forecast_horizon=model_config.get("forecast_horizon", 1),
        warmup_period=model_config.get("warmup_period", 60),
        fit_interval=model_config.get("fit_interval", 0),
    )

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

    mae_dict = mae_analysis.get("mean_absolute_error", {})
    data_name = list(mae_dict.keys())[0] if mae_dict else ""

    if generate_plot and plot_path:
        generate_walkforward_plot(strat, test_df, ticker, model_name, plot_path)

    return {
        "total_return": total_return,
        "bh_return": bh_return,
        "excess_return": total_return - bh_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "mae": mae_dict.get(data_name, -1) if data_name else -1,
        "rmse": rmse_analysis.get("root_mean_squared_error", {}).get(data_name, -1) if data_name else -1,
        "mape": mape_analysis.get("mape", {}).get(data_name, -1) if data_name else -1,
        "oos_r2": oos_r2_analysis.get("oos_r_squared", {}).get(data_name, -1) if data_name else -1,
        "directional_accuracy": da_analysis.get("directional_accuracy", {}).get(data_name, -1) if data_name else -1,
    }


def generate_walkforward_plot(strat, test_df, ticker, model_name, plot_path):
    """Generate a walk-forward visualization plot."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [1, 2]})

    dates = test_df.index
    closes = test_df["Close"].values

    d = strat.datas[0]
    predictions = strat.predictions.get(d, [])
    prices = strat.prices.get(d, [])

    portfolio_values = []
    current_value = INITIAL_CASH
    for i, price in enumerate(prices):
        if i < len(prices) - 1:
            if i < len(predictions) and predictions[i] is not None:
                pred_return = np.log(predictions[i] / price) if predictions[i] and price > 0 else 0
                actual_return = np.log(prices[i+1] / price) if prices[i+1] and price > 0 else 0
                current_value *= (1 + actual_return * 0.5)
        portfolio_values.append(current_value)

    if len(portfolio_values) < len(dates):
        portfolio_values.extend([portfolio_values[-1]] * (len(dates) - len(portfolio_values)))
    portfolio_values = portfolio_values[:len(dates)]

    ax1.plot(dates, portfolio_values, 'b-', linewidth=1.5, label='Portfolio')
    ax1.axhline(y=INITIAL_CASH, color='gray', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Portfoliowert ($)')
    ax1.set_title(f'Walk-Forward Backtest: {ticker} - {model_name}')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    ax2.plot(dates, closes, 'k-', linewidth=1.5, label=f'{ticker} Kurs')

    pred_dates = []
    pred_values = []
    for i, pred in enumerate(predictions):
        if pred is not None and i < len(dates):
            pred_dates.append(dates[i])
            pred_values.append(pred)

    if pred_dates:
        ax2.scatter(pred_dates, pred_values, c='blue', s=10, alpha=0.5, label='Vorhersagen')

    buy_dates = []
    buy_prices = []
    sell_dates = []
    sell_prices = []

    predicted_returns = strat.predicted_returns.get(d, [])
    threshold = strat.p.signal_threshold

    for i, pred_ret in enumerate(predicted_returns):
        if pred_ret is not None and i < len(dates) and i < len(closes):
            if pred_ret > threshold:
                buy_dates.append(dates[i])
                buy_prices.append(closes[i])
            elif pred_ret < -threshold:
                sell_dates.append(dates[i])
                sell_prices.append(closes[i])

    if buy_dates:
        ax2.scatter(buy_dates, buy_prices, marker='^', c='green', s=50, label='Kaufsignal', zorder=5)
    if sell_dates:
        ax2.scatter(sell_dates, sell_prices, marker='v', c='red', s=50, label='Verkaufsignal', zorder=5)

    ax2.set_xlabel('Datum')
    ax2.set_ylabel('Kurs ($)')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Walk-Forward Model Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run python run_evaluation.py                          # All models
    uv run python run_evaluation.py --models LinearSeparation
    uv run python run_evaluation.py --models LinearSeparation SVR
    uv run python run_evaluation.py --list-models
        """
    )
    parser.add_argument(
        "--models", "-m",
        nargs="+",
        help="Models to run (default: all available)",
        metavar="MODEL"
    )
    parser.add_argument(
        "--list-models", "-l",
        action="store_true",
        help="List available models and exit"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list_models:
        print("Available models:")
        for name in MODEL_CONFIGS.keys():
            print(f"  - {name}")
        return

    config = load_config()
    stocks = get_selected_stocks(config)

    if args.models:
        invalid = [m for m in args.models if m not in MODEL_CONFIGS]
        if invalid:
            print(f"ERROR: Unknown model(s): {invalid}")
            print(f"Available: {list(MODEL_CONFIGS.keys())}")
            sys.exit(1)
        selected_models = {k: v for k, v in MODEL_CONFIGS.items() if k in args.models}
    else:
        selected_models = MODEL_CONFIGS

    print("=" * 80)
    print("WALK-FORWARD MODEL EVALUATION")
    print("=" * 80)
    print(f"Stocks: {stocks}")
    print(f"Models: {list(selected_models.keys())}")
    print(f"Initial Cash: ${INITIAL_CASH:,.0f}")
    print()

    all_results = []
    example_plot_generated = False

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

        sentiment_data = get_sentiment_for_model(ticker)
        if len(sentiment_data) == 0:
            all_dates = pd.concat([train_df, val_df, test_df]).index
            sentiment_data = generate_placeholder_sentiment(all_dates, seed=hash(ticker) % 2**32)

        for model_name, model_config in selected_models.items():
            print(f"  Running {model_name}...", end=" ")

            try:
                if model_name == "SVR":
                    model = model_config["train_fn"](train_df, val_df, sentiment_data)
                else:
                    model = model_config["train_fn"](train_df, val_df)

                if model is None:
                    print("SKIPPED (model not available)")
                    continue

                signal_threshold = compute_validation_mae(model, val_df)
                print(f"ε={signal_threshold:.4f}", end=" ")

                config_with_threshold = model_config.copy()
                config_with_threshold["signal_threshold"] = signal_threshold

                generate_plot = not example_plot_generated and model_name == "LinearSeparation"
                plot_path = os.path.join(OUTPUT_DIR, f"walkforward_example_{ticker}_{model_name}.png") if generate_plot else None

                metrics = run_backtest(
                    model, test_df, config_with_threshold,
                    generate_plot=generate_plot,
                    plot_path=plot_path,
                    ticker=ticker,
                    model_name=model_name
                )

                if generate_plot:
                    example_plot_generated = True
                    print(f"(plot saved)", end=" ")

                metrics["ticker"] = ticker
                metrics["model"] = model_name
                metrics["epsilon"] = signal_threshold
                all_results.append(metrics)

                print(f"Return: {metrics['total_return']:.2%}, "
                      f"DA: {metrics['directional_accuracy']:.3f}, "
                      f"OOS-R²: {metrics['oos_r2']:.4f}")

            except Exception as e:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()
                continue

    if all_results:
        print("\n\n")
        print("=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)

        df = pd.DataFrame(all_results)
        cols = ["ticker", "model", "epsilon", "total_return", "bh_return", "excess_return",
                "sharpe_ratio", "max_drawdown", "mape", "oos_r2", "directional_accuracy"]
        print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        print("\n\nAverage by Model:")
        numeric_cols = ["total_return", "excess_return", "sharpe_ratio",
                        "max_drawdown", "mape", "oos_r2", "directional_accuracy"]
        print(df.groupby("model")[numeric_cols].mean().to_string(float_format=lambda x: f"{x:.4f}"))

        results_json = {
            "config": {
                "initial_cash": INITIAL_CASH,
                "stocks": stocks,
                "models": list(selected_models.keys()),
            },
            "results": all_results,
            "summary_by_model": df.groupby("model")[numeric_cols].mean().to_dict(),
        }

        json_path = os.path.join(OUTPUT_DIR, "evaluation_results.json")
        with open(json_path, "w") as f:
            json.dump(results_json, f, indent=2, default=str)
        print(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    main()
