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

    # Check if model supports incremental updates (like ARIMA)
    has_update = hasattr(model, 'update')

    # For SVR: need to update data_buffer with each new row
    is_svr = hasattr(model, 'data_buffer') and hasattr(model, 'append')

    for i in range(30, len(closes) - 1):
        try:
            # For SVR models, update data buffer with current validation data
            if is_svr:
                # Set data_buffer to validation data up to current point
                current_val_data = val_df.iloc[:i+1].copy()
                # Rename columns to lowercase as SVR expects
                current_val_data.columns = current_val_data.columns.str.lower()
                model.data_buffer = current_val_data
                model.last_close = closes[i]

            prediction = model.predict(n=1)
            if prediction is None or np.isnan(prediction):
                if has_update:
                    model.update(closes[i])
                continue

            actual_return = np.log(closes[i + 1] / closes[i])
            predicted_return = np.log(prediction / closes[i])
            errors.append(abs(predicted_return - actual_return))

            if has_update:
                model.update(closes[i])
        except Exception as e:
            if has_update:
                model.update(closes[i])
            continue

    if len(errors) == 0:
        return 0.005

    mae = np.mean(errors)
    return mae * 0.3


def train_linear_separation(train_df, val_df):
    """Train LinearSeparation model (uses ARIMA internally)."""
    model = ARIMAModel(order=(2, 1, 1), window=252)
    model.fit(train_df["Close"].values)
    return model


def train_svr(train_df, val_df, sentiment_data=None):
    """Train SVR model with grid search on validation data.

    Uses 5-day prediction horizon for better signal quality.
    """
    model = SVRPredictionModel(prediction_horizon=5)
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
    try:
        from models.timesfm_prediction_model import TimesFMPredictionModel
    except ImportError:
        print("ERROR: TimesFM nicht installiert.")
        return None
    model = TimesFMPredictionModel()
    model.fit(train_df)
    return model


def is_timesfm_available():
    """Check if TimesFM is available without loading the model."""
    try:
        import importlib.util
        spec = importlib.util.find_spec("timesfm")
        return spec is not None
    except Exception:
        return False


MODEL_CONFIGS = {
    "LinearSeparation": {
        "train_fn": train_linear_separation,
        "forecast_horizon": 1,
        "warmup_period": 0,
    },
    "SVR": {
        "train_fn": train_svr,
        "forecast_horizon": 5,  # Predict 5 days ahead
        "warmup_period": 0,
    },
}

if is_timesfm_available():
    MODEL_CONFIGS["TimesFM"] = {
        "train_fn": train_timesfm,
        "forecast_horizon": 1,
        "warmup_period": 0,
    }


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

    # Extract trade statistics from strategy
    d = strat.datas[0]
    trade_stats = {
        "buy_orders": strat.buy_count.get(d, 0),
        "sell_orders": strat.sell_count.get(d, 0),
        "close_positions": strat.close_count.get(d, 0),
        "total_trades": strat.buy_count.get(d, 0) + strat.sell_count.get(d, 0) + strat.close_count.get(d, 0)
    }

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
        "trade_statistics": trade_stats,
    }


def generate_walkforward_plot(strat, test_df, ticker, model_name, plot_path):
    """Generate a walk-forward visualization plot."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [1, 2]})

    dates = test_df.index
    closes = test_df["Close"].values

    d = strat.datas[0]
    predictions = strat.predictions.get(d, [])
    prices = strat.prices.get(d, [])
    signals = strat.signals.get(d, [])

    portfolio_values = []
    current_value = INITIAL_CASH
    position = 0

    for i, price in enumerate(prices):
        if i < len(signals):
            sig = signals[i]
            if sig == 1 and position == 0:
                position = current_value * 0.5 / price
                current_value -= position * price
            elif sig == -1 and position > 0:
                current_value += position * price
                position = 0

        total_value = current_value + position * price
        portfolio_values.append(total_value)

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

    signals = strat.signals.get(d, [])

    for i, sig in enumerate(signals):
        if i < len(dates) and i < len(closes):
            if sig == 1:
                buy_dates.append(dates[i])
                buy_prices.append(closes[i])
            elif sig == -1:
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

    # Store results per model
    results_by_model = {}

    for model_name in selected_models.keys():
        results_by_model[model_name] = []

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



        for model_name, model_config in selected_models.items():
            print(f"  Running {model_name}...", end=" ")
            try:
                if model_name == "SVR":
                    sentiment_data = get_sentiment_for_model(ticker)
                    if len(sentiment_data) == 0:
                        all_dates = pd.concat([train_df, val_df, test_df]).index
                        sentiment_data = generate_placeholder_sentiment(all_dates, seed=hash(ticker) % 2**32)
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

                # Create model-specific output directory
                model_output_dir = os.path.join(OUTPUT_DIR, model_name)
                os.makedirs(model_output_dir, exist_ok=True)

                # Generate plots for all stocks
                generate_plot = True
                plot_path = os.path.join(model_output_dir, f"walkforward_{ticker}_{model_name}.png")

                metrics = run_backtest(
                    model, test_df, config_with_threshold,
                    generate_plot=generate_plot,
                    plot_path=plot_path,
                    ticker=ticker,
                    model_name=model_name
                )

                if generate_plot:
                    print(f"(plot saved)", end=" ")

                metrics["ticker"] = ticker
                metrics["model"] = model_name
                metrics["epsilon"] = signal_threshold

                # Add model-specific hyperparameters
                if model_name == "SVR" and hasattr(model, 'C'):
                    metrics["hyperparameters"] = {
                        "C": model.C,
                        "epsilon": model.epsilon,
                        "gamma": model.gamma,
                        "lookback_window": model.lookback_window,
                        "prediction_horizon": model.prediction_horizon
                    }
                elif model_name == "ARIMA" and hasattr(model, 'order'):
                    metrics["hyperparameters"] = {
                        "order": model.order,
                        "window": getattr(model, 'window', None)
                    }
                elif model_name == "LinearSeparation" and hasattr(model, 'order'):
                    metrics["hyperparameters"] = {
                        "order": model.order,
                        "window": getattr(model, 'window', None)
                    }

                results_by_model[model_name].append(metrics)

                trade_stats = metrics.get("trade_statistics", {})
                print(f"Return: {metrics['total_return']:.2%}, "
                      f"DA: {metrics['directional_accuracy']:.3f}, "
                      f"OOS-R²: {metrics['oos_r2']:.4f}, "
                      f"Trades: {trade_stats.get('total_trades', 0)} "
                      f"(B:{trade_stats.get('buy_orders', 0)} S:{trade_stats.get('sell_orders', 0)} C:{trade_stats.get('close_positions', 0)})")

            except Exception as e:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()
                continue

    # Save results per model
    for model_name, model_results in results_by_model.items():
        if not model_results:
            continue

        print("\n\n")
        print("=" * 80)
        print(f"RESULTS SUMMARY - {model_name}")
        print("=" * 80)

        df = pd.DataFrame(model_results)
        cols = ["ticker", "model", "epsilon", "total_return", "bh_return", "excess_return",
                "sharpe_ratio", "max_drawdown", "mape", "oos_r2", "directional_accuracy"]
        print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        print("\n\nAverage Metrics:")
        numeric_cols = ["total_return", "excess_return", "sharpe_ratio",
                        "max_drawdown", "mape", "oos_r2", "directional_accuracy"]
        avg_metrics = df[numeric_cols].mean()
        for col in numeric_cols:
            print(f"  {col}: {avg_metrics[col]:.4f}")

        # Trade statistics summary
        print("\n\nTrade Statistics:")
        total_buys = sum(r["trade_statistics"]["buy_orders"] for r in model_results)
        total_sells = sum(r["trade_statistics"]["sell_orders"] for r in model_results)
        total_closes = sum(r["trade_statistics"]["close_positions"] for r in model_results)
        total_trades = sum(r["trade_statistics"]["total_trades"] for r in model_results)
        print(f"  Total Trades: {total_trades}")
        if total_trades > 0:
            print(f"  Buy Orders: {total_buys} ({total_buys/total_trades*100:.1f}%)")
            print(f"  Sell Orders: {total_sells} ({total_sells/total_trades*100:.1f}%)")
            print(f"  Close Positions: {total_closes} ({total_closes/total_trades*100:.1f}%)")
            print(f"  Average per Stock: {total_trades/len(model_results):.1f}")
        else:
            print(f"  WARNING: No trades executed! Model may not be generating valid predictions.")
            print(f"  Buy Orders: {total_buys}")
            print(f"  Sell Orders: {total_sells}")
            print(f"  Close Positions: {total_closes}")

        # Save to model-specific directory
        model_output_dir = os.path.join(OUTPUT_DIR, model_name)
        os.makedirs(model_output_dir, exist_ok=True)

        results_json = {
            "config": {
                "initial_cash": INITIAL_CASH,
                "stocks": stocks,
                "model": model_name,
                "data_split": {
                    "training_start": config.get("training_start"),
                    "training_end": config.get("training_end"),
                    "validation_start": config.get("validation_start"),
                    "validation_end": config.get("validation_end"),
                    "test_start": config.get("test_start"),
                    "test_end": config.get("test_end")
                },
                "model_config": MODEL_CONFIGS.get(model_name, {})
            },
            "results": model_results,
            "summary": {
                "metrics": avg_metrics.to_dict(),
                "trade_statistics": {
                    "total_trades": total_trades,
                    "buy_orders": total_buys,
                    "sell_orders": total_sells,
                    "close_positions": total_closes,
                    "average_per_stock": total_trades / len(model_results)
                }
            }
        }

        json_path = os.path.join(model_output_dir, "evaluation_results.json")
        with open(json_path, "w") as f:
            json.dump(results_json, f, indent=2, default=str)
        print(f"\nResults saved to {json_path}")

        # Save trade statistics CSV
        trade_stats_rows = []
        for result in model_results:
            ts = result["trade_statistics"]
            row = {
                "Stock": result["ticker"],
                "Model": model_name,
                "Signal_Threshold_Epsilon": result["epsilon"],
                "Buy_Orders": ts["buy_orders"],
                "Sell_Orders": ts["sell_orders"],
                "Close_Positions": ts["close_positions"],
                "Total_Trades": ts["total_trades"],
                "Total_Return": result["total_return"],
                "BuyHold_Return": result["bh_return"],
                "Excess_Return": result["excess_return"],
                "Sharpe_Ratio": result["sharpe_ratio"],
                "Max_Drawdown": result["max_drawdown"],
                "MAE": result["mae"],
                "RMSE": result["rmse"],
                "MAPE": result["mape"],
                "Directional_Accuracy": result["directional_accuracy"],
                "OOS_R_Squared": result["oos_r2"]
            }

            # Add hyperparameters if available
            if "hyperparameters" in result:
                for key, value in result["hyperparameters"].items():
                    row[f"HP_{key}"] = value

            trade_stats_rows.append(row)

        trade_df = pd.DataFrame(trade_stats_rows)
        csv_path = os.path.join(model_output_dir, "trade_statistics.csv")
        trade_df.to_csv(csv_path, index=False)
        print(f"Trade statistics saved to {csv_path}")

        # Save detailed markdown summary
        md_path = os.path.join(model_output_dir, "evaluation_summary.md")
        with open(md_path, "w") as f:
            f.write(f"# {model_name} Model Evaluation Summary\n\n")
            f.write(f"**Evaluation Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## Configuration\n\n")
            f.write(f"- **Model**: {model_name}\n")
            f.write(f"- **Initial Cash**: ${INITIAL_CASH:,.0f}\n")
            f.write(f"- **Number of Stocks**: {len(stocks)}\n")
            f.write(f"- **Stocks**: {', '.join(stocks)}\n\n")

            f.write("### Data Split\n\n")
            f.write(f"- **Training**: {config.get('training_start')} to {config.get('training_end')}\n")
            f.write(f"- **Validation**: {config.get('validation_start')} to {config.get('validation_end')}\n")
            f.write(f"- **Test**: {config.get('test_start')} to {config.get('test_end')}\n\n")

            f.write("## Average Performance Metrics\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            f.write(f"| Total Return | {avg_metrics['total_return']:.2%} |\n")
            f.write(f"| Excess Return | {avg_metrics['excess_return']:.2%} |\n")
            f.write(f"| Sharpe Ratio | {avg_metrics['sharpe_ratio']:.4f} |\n")
            f.write(f"| Max Drawdown | {avg_metrics['max_drawdown']:.2%} |\n")
            f.write(f"| MAPE | {avg_metrics['mape']:.4f} |\n")
            f.write(f"| RMSE | {df['rmse'].mean():.4f} |\n")
            f.write(f"| MAE | {df['mae'].mean():.4f} |\n")
            f.write(f"| OOS R² | {avg_metrics['oos_r2']:.4f} |\n")
            f.write(f"| Directional Accuracy | {avg_metrics['directional_accuracy']:.2%} |\n\n")

            f.write("## Trade Statistics\n\n")
            f.write(f"- **Total Trades**: {total_trades}\n")
            f.write(f"- **Buy Orders**: {total_buys} ({total_buys/total_trades*100:.1f}%)\n")
            f.write(f"- **Sell Orders**: {total_sells} ({total_sells/total_trades*100:.1f}%)\n")
            f.write(f"- **Close Positions**: {total_closes} ({total_closes/total_trades*100:.1f}%)\n")
            f.write(f"- **Average per Stock**: {total_trades/len(model_results):.1f}\n\n")

            f.write("## Results by Stock\n\n")
            f.write("| Stock | Total Return | Excess Return | Sharpe | Max DD | DA | OOS R² | Trades | ε |\n")
            f.write("|-------|--------------|---------------|--------|--------|-------|--------|--------|----|\n")
            for result in model_results:
                ts = result["trade_statistics"]
                f.write(f"| {result['ticker']} | {result['total_return']:.2%} | "
                       f"{result['excess_return']:.2%} | {result['sharpe_ratio']:.2f} | "
                       f"{result['max_drawdown']:.2%} | {result['directional_accuracy']:.2%} | "
                       f"{result['oos_r2']:.3f} | {ts['total_trades']} | {result['epsilon']:.4f} |\n")

            f.write("\n## Best Performers\n\n")
            sorted_by_return = sorted(model_results, key=lambda x: x['total_return'], reverse=True)[:3]
            f.write("### Top 3 by Total Return\n\n")
            for i, result in enumerate(sorted_by_return, 1):
                f.write(f"{i}. **{result['ticker']}**: {result['total_return']:.2%} "
                       f"(Excess: {result['excess_return']:.2%})\n")

            f.write("\n### Top 3 by Sharpe Ratio\n\n")
            sorted_by_sharpe = sorted(model_results, key=lambda x: x['sharpe_ratio'] if x['sharpe_ratio'] else -999, reverse=True)[:3]
            for i, result in enumerate(sorted_by_sharpe, 1):
                sharpe_str = f"{result['sharpe_ratio']:.2f}" if result['sharpe_ratio'] else "N/A"
                f.write(f"{i}. **{result['ticker']}**: Sharpe {sharpe_str} "
                       f"(Return: {result['total_return']:.2%})\n")

            f.write("\n### Top 3 by Directional Accuracy\n\n")
            sorted_by_da = sorted(model_results, key=lambda x: x['directional_accuracy'], reverse=True)[:3]
            for i, result in enumerate(sorted_by_da, 1):
                f.write(f"{i}. **{result['ticker']}**: {result['directional_accuracy']:.2%} "
                       f"(Return: {result['total_return']:.2%})\n")

            if any("hyperparameters" in r for r in model_results):
                f.write("\n## Hyperparameters by Stock\n\n")
                hp_results = [r for r in model_results if "hyperparameters" in r]
                if hp_results:
                    hp_keys = list(hp_results[0]["hyperparameters"].keys())
                    header = "| Stock | " + " | ".join(hp_keys) + " |\n"
                    separator = "|-------|" + "|".join(["-------"] * len(hp_keys)) + "|\n"
                    f.write(header)
                    f.write(separator)
                    for result in hp_results:
                        hp_values = [str(result["hyperparameters"][k]) for k in hp_keys]
                        f.write(f"| {result['ticker']} | " + " | ".join(hp_values) + " |\n")

        print(f"Markdown summary saved to {md_path}")


if __name__ == "__main__":
    main()
