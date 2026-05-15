#!/usr/bin/env python3
"""Extract and analyze trade statistics from evaluation results."""

import json
import os
import pandas as pd

def load_results(json_path="data/evaluation/evaluation_results.json"):
    """Load evaluation results from JSON."""
    with open(json_path, "r") as f:
        return json.load(f)

def extract_trade_stats(results):
    """Extract trade statistics into a DataFrame."""
    rows = []
    for result in results["results"]:
        trade_stats = result.get("trade_statistics", {})
        rows.append({
            "Stock": result["ticker"],
            "Model": result["model"],
            "Buy Orders": trade_stats.get("buy_orders", 0),
            "Sell Orders": trade_stats.get("sell_orders", 0),
            "Close Positions": trade_stats.get("close_positions", 0),
            "Total Trades": trade_stats.get("total_trades", 0),
            "Total Return": result["total_return"],
            "Directional Accuracy": result["directional_accuracy"],
            "OOS-R²": result["oos_r2"],
        })
    return pd.DataFrame(rows)

def analyze_trades(df):
    """Generate summary statistics."""
    print("\n" + "=" * 80)
    print("TRADE STATISTICS ANALYSIS")
    print("=" * 80)

    print("\n📊 Trade Statistics by Stock:")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n\n📈 Summary Statistics:")
    print(f"  Total Trades (all stocks): {df['Total Trades'].sum()}")
    print(f"  Average Trades per Stock: {df['Total Trades'].mean():.1f}")
    print(f"  Min Trades: {df['Total Trades'].min()} ({df.loc[df['Total Trades'].idxmin(), 'Stock']})")
    print(f"  Max Trades: {df['Total Trades'].max()} ({df.loc[df['Total Trades'].idxmax(), 'Stock']})")

    print(f"\n  Total Buy Orders: {df['Buy Orders'].sum()}")
    print(f"  Total Sell Orders: {df['Sell Orders'].sum()}")
    print(f"  Total Close Positions: {df['Close Positions'].sum()}")

    print(f"\n  Average Buy Orders per Stock: {df['Buy Orders'].mean():.1f}")
    print(f"  Average Sell Orders per Stock: {df['Sell Orders'].mean():.1f}")
    print(f"  Average Close Positions per Stock: {df['Close Positions'].mean():.1f}")

    # Correlation analysis
    print("\n\n📉 Correlation with Performance:")
    corr_return = df[['Total Trades', 'Total Return']].corr().iloc[0, 1]
    corr_da = df[['Total Trades', 'Directional Accuracy']].corr().iloc[0, 1]
    corr_oos = df[['Total Trades', 'OOS-R²']].corr().iloc[0, 1]

    print(f"  Total Trades vs. Total Return: {corr_return:.3f}")
    print(f"  Total Trades vs. Directional Accuracy: {corr_da:.3f}")
    print(f"  Total Trades vs. OOS-R²: {corr_oos:.3f}")

    # Trade frequency analysis
    test_period = 377  # days
    print(f"\n\n⏱️  Trade Frequency Analysis (Test Period: {test_period} days):")
    avg_days_per_trade = test_period / df['Total Trades'].mean()
    print(f"  Average days between trades: {avg_days_per_trade:.1f}")

    signal_cooldown = 5
    theoretical_max = test_period / signal_cooldown
    utilization = (df['Total Trades'].mean() / theoretical_max) * 100
    print(f"  Theoretical max trades (with cooldown={signal_cooldown}): {theoretical_max:.0f}")
    print(f"  Actual utilization: {utilization:.1f}%")

    # Best and worst performers
    print("\n\n🏆 Performance Ranking:")
    sorted_df = df.sort_values("Total Return", ascending=False)
    print("\nTop 3 Performers:")
    for idx, row in sorted_df.head(3).iterrows():
        print(f"  {row['Stock']}: {row['Total Return']:.2%} return, {row['Total Trades']} trades")

    print("\nBottom 3 Performers:")
    for idx, row in sorted_df.tail(3).iterrows():
        print(f"  {row['Stock']}: {row['Total Return']:.2%} return, {row['Total Trades']} trades")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, "data", "evaluation", "evaluation_results.json")

    if not os.path.exists(json_path):
        print(f"ERROR: {json_path} not found. Run run_evaluation.py first.")
        return

    results = load_results(json_path)
    df = extract_trade_stats(results)
    analyze_trades(df)

    # Save to CSV
    csv_path = os.path.join(script_dir, "data", "evaluation", "trade_statistics.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n\n💾 Trade statistics saved to: {csv_path}")

if __name__ == "__main__":
    main()
