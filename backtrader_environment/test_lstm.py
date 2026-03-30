"""
LSTM Stock Prediction Backtest

Tests the Multi-Feature Attention LSTM strategy on individual assets with:
- Return prediction (not price prediction)
- 70%/30% train/test split
- No refitting during test period
- $100k initial capital per asset
"""

import warnings
warnings.filterwarnings('ignore')

import backtrader as bt
import pandas as pd
import yaml
import os

from strategies import LSTMStrategy
from analyzer import (
    MeanAbsoluteErrorAnalyzer,
    RootMeanSquaredErrorAnalyzer,
    MeanSquaredErrorAnalyzer,
    RSquaredAnalyzer
)
from sizing import PercentageSizer


def load_config():
    """Load configuration from config.yaml."""
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_csv_data(symbol, data_dir='data/assets'):
    """Load CSV data as DataFrame."""
    csv_path = os.path.join(os.path.dirname(__file__), data_dir, f'{symbol}.csv')
    if not os.path.exists(csv_path):
        return None
    return pd.read_csv(csv_path, parse_dates=['Date'], index_col='Date')


def run_backtest(symbol, df, strategy_params, train_ratio=0.7,
                 initial_cash=100000.0, commission=0.0):
    """
    Run LSTM backtest on a single asset with train/test split.

    IMPORTANT: Model is pre-trained on training data BEFORE backtest starts.
    Backtest runs ONLY on test data for fair comparison with buy-and-hold.
    """
    # Split data
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    # Calculate buy-and-hold return for test period
    bh_return = ((test_df['Close'].iloc[-1] - test_df['Close'].iloc[0]) / test_df['Close'].iloc[0]) * 100

    # Pre-train the LSTM model on training data
    from models.lstm_prediction_model import LSTMPredictionModel

    pretrained_model = LSTMPredictionModel(
        lookback_window=strategy_params.get('lookback_window', 60),
        forecast_horizon=strategy_params.get('forecast_horizon', 5),
        hidden_size_1=strategy_params.get('hidden_size_1', 64),
        hidden_size_2=strategy_params.get('hidden_size_2', 32),
        num_heads=strategy_params.get('num_heads', 4),
        dropout=strategy_params.get('dropout', 0.2),
        dense_units=strategy_params.get('dense_units', 64),
        epochs=strategy_params.get('epochs', 50),
        batch_size=strategy_params.get('batch_size', 32),
        learning_rate=strategy_params.get('learning_rate', 0.001),
        early_stopping_patience=strategy_params.get('early_stopping_patience', 5)
    )

    # Train on training data only
    train_df_lower = train_df.copy()
    train_df_lower.columns = [c.lower() for c in train_df_lower.columns]
    pretrained_model.fit(train_df_lower)

    # Pass pre-trained model to strategy
    strategy_params['pretrained_model'] = pretrained_model
    strategy_params['fit_interval'] = 0  # No refitting

    # Create Cerebro with standard stats for plotting
    cerebro = bt.Cerebro(stdstats=True)
    cerebro.addstrategy(LSTMStrategy, **strategy_params)

    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.02)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(MeanAbsoluteErrorAnalyzer, _name='mae')
    cerebro.addanalyzer(MeanSquaredErrorAnalyzer, _name='mse')
    cerebro.addanalyzer(RootMeanSquaredErrorAnalyzer, _name='rmse')
    cerebro.addanalyzer(RSquaredAnalyzer, _name='r2')
    cerebro.addsizer(PercentageSizer)

    # Add ONLY TEST DATA to backtest (fair comparison with buy-and-hold)
    data = bt.feeds.PandasData(
        dataname=test_df, datetime=None,
        open='Open', high='High', low='Low', close='Close', volume='Volume',
        openinterest=-1
    )
    cerebro.adddata(data, name=symbol)

    # Broker settings
    cerebro.broker.set_cash(initial_cash)
    cerebro.broker.setcommission(commission=commission)

    # Run
    results = cerebro.run()
    strat = results[0]

    # Extract results
    final_value = cerebro.broker.getvalue()
    total_return = ((final_value - initial_cash) / initial_cash) * 100

    sharpe = strat.analyzers.sharpe.get_analysis()
    drawdown = strat.analyzers.drawdown.get_analysis()
    trades = strat.analyzers.trades.get_analysis()
    mae = strat.analyzers.mae.get_analysis()
    mse = strat.analyzers.mse.get_analysis()
    rmse = strat.analyzers.rmse.get_analysis()
    r2 = strat.analyzers.r2.get_analysis()

    return {
        'symbol': symbol,
        'train_start': train_df.index[0].strftime('%Y-%m-%d'),
        'train_end': train_df.index[-1].strftime('%Y-%m-%d'),
        'test_start': test_df.index[0].strftime('%Y-%m-%d'),
        'test_end': test_df.index[-1].strftime('%Y-%m-%d'),
        'train_days': len(train_df),
        'test_days': len(test_df),
        'total_return': total_return,
        'bh_return': bh_return,
        'excess_return': total_return - bh_return,
        'final_value': final_value,
        'sharpe': sharpe.get('sharperatio'),
        'max_drawdown': drawdown.get('max', {}).get('drawdown', 0),
        'total_trades': trades.get('total', {}).get('total', 0),
        'won_trades': trades.get('won', {}).get('total', 0),
        'lost_trades': trades.get('lost', {}).get('total', 0),
        'mae': mae.get('mean_absolute_error', {}).get(symbol, -1),
        'mse': mse.get('mean_squared_error', {}).get(symbol, -1),
        'rmse': rmse.get('root_mean_squared_error', {}).get(symbol, -1),
        'r2': r2.get('r_squared_analysis', {}).get(symbol, -1),
        'cerebro': cerebro,  # Return cerebro for plotting
    }


def print_results(results):
    """Print detailed results for all assets."""
    print("\n" + "=" * 80)
    print("DETAILED RESULTS")
    print("=" * 80)

    for r in results:
        print(f"\n[{r['symbol']}]")
        print(f"  Train: {r['train_start']} to {r['train_end']} ({r['train_days']} days)")
        print(f"  Test:  {r['test_start']} to {r['test_end']} ({r['test_days']} days)")
        print(f"  ---")
        print(f"  Final Value:    ${r['final_value']:,.2f}")
        print(f"  Total Return:   {r['total_return']:+.2f}%")
        print(f"  Buy & Hold:     {r['bh_return']:+.2f}%")
        print(f"  Excess Return:  {r['excess_return']:+.2f}%")
        print(f"  ---")
        print(f"  Sharpe Ratio:   {r['sharpe']:.4f}" if r['sharpe'] else "  Sharpe Ratio:   N/A")
        print(f"  Max Drawdown:   {r['max_drawdown']:.2f}%")
        print(f"  Total Trades:   {r['total_trades']}")
        print(f"  Won/Lost:       {r['won_trades']}/{r['lost_trades']}")
        if r['total_trades'] > 0:
            print(f"  Win Rate:       {r['won_trades']/r['total_trades']*100:.1f}%")
        print(f"  ---")
        print(f"  MAE:            {r['mae']:.4f}" if r['mae'] != -1 else "  MAE:            N/A")
        print(f"  MSE:            {r['mse']:.4f}" if r['mse'] != -1 else "  MSE:            N/A")
        print(f"  RMSE:           {r['rmse']:.4f}" if r['rmse'] != -1 else "  RMSE:           N/A")
        print(f"  R²:             {r['r2']:.4f}" if r['r2'] != -1 else "  R²:             N/A")

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"\n{'Asset':<8} {'Return':>10} {'B&H':>10} {'Excess':>10} {'Sharpe':>10} {'R²':>10} {'Trades':>8}")
    print("-" * 80)

    total_ret = 0
    total_bh = 0

    for r in results:
        sharpe_str = f"{r['sharpe']:.4f}" if r['sharpe'] else "N/A"
        r2_str = f"{r['r2']:.4f}" if r['r2'] != -1 else "N/A"

        print(f"{r['symbol']:<8} {r['total_return']:>9.2f}% {r['bh_return']:>9.2f}% "
              f"{r['excess_return']:>+9.2f}% {sharpe_str:>10} {r2_str:>10} {r['total_trades']:>8}")

        total_ret += r['total_return']
        total_bh += r['bh_return']

    n = len(results)
    print("-" * 80)
    print(f"{'AVERAGE':<8} {total_ret/n:>9.2f}% {total_bh/n:>9.2f}% {(total_ret-total_bh)/n:>+9.2f}%")

    print("\n" + "=" * 80)


def main():
    print("=" * 80)
    print("LSTM STOCK PREDICTION BACKTEST")
    print("Multi-Feature Attention LSTM | Return Prediction | No Refitting")
    print("=" * 80)

    # Load config
    config = load_config()
    assets = config.get('assets', ['AAPL', 'MSFT'])

    print(f"\nAssets: {', '.join(assets)}")
    print(f"Train/Test Split: 70%/30%")
    print(f"Initial Cash: $100,000 per asset")
    print(f"Commission: 0.0%")

    # LSTM parameters
    lstm_params = {
        'lookback_window': 30,
        'forecast_horizon': 5,   # Predict 5-day ahead return
        'hidden_size_1': 16,
        'hidden_size_2': 8,
        'num_heads': 2,
        'epochs': 15,
        'batch_size': 16,
        'warmup_period': 30,     # Same as lookback_window since model is pre-trained
        'signal_threshold': 0.01, # 1% threshold
    }

    print(f"\nLSTM Parameters:")
    print(f"  Lookback Window: {lstm_params['lookback_window']}")
    print(f"  Forecast Horizon: {lstm_params['forecast_horizon']} days")
    print(f"  Hidden Layers:   {lstm_params['hidden_size_1']} -> {lstm_params['hidden_size_2']}")
    print(f"  Attention Heads: {lstm_params['num_heads']}")
    print(f"  Training Epochs: {lstm_params['epochs']}")
    print(f"  Signal Threshold: {lstm_params['signal_threshold']*100:.1f}%")

    # Run backtests
    print("\n" + "-" * 80)
    print("Running backtests...")
    print("-" * 80)

    results = []
    for symbol in assets:
        df = load_csv_data(symbol)
        if df is None:
            print(f"[{symbol}] Data not found, skipping")
            continue

        print(f"[{symbol}] Training and testing...", end=" ", flush=True)
        result = run_backtest(symbol, df, lstm_params.copy())
        results.append(result)
        print(f"Return: {result['total_return']:+.2f}%")

    if results:
        print_results(results)

        # Plot each asset separately using Backtrader's plotting
        print("\nGenerating plots for each asset...")
        for result in results:
            print(f"  Plotting {result['symbol']}...")
            result['cerebro'].plot(
                style='line',
                barup='green',
                bardown='red',
                volume=True,
                figsize=(16, 10)
            )
    else:
        print("No results to display.")


if __name__ == '__main__':
    main()
