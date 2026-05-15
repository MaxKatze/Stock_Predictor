# SVR Model Evaluation Summary

**Evaluation Date**: 2026-05-15 12:06:21

## Configuration

- **Model**: SVR
- **Initial Cash**: $100,000
- **Number of Stocks**: 10
- **Stocks**: CARR, OTIS, RSG, CTAS, AJG, PEP, TT, GWW, RMD, BLK

### Data Split

- **Training**: 2020-01-02 to 2023-12-29
- **Validation**: 2024-01-02 to 2024-06-28
- **Test**: 2024-07-01 to 2025-12-31

## Average Performance Metrics

| Metric | Value |
|--------|-------|
| Total Return | 6.56% |
| Excess Return | -5.05% |
| Sharpe Ratio | 0.2045 |
| Max Drawdown | 13.95% |
| MAPE | 0.0130 |
| RMSE | 7.0968 |
| MAE | 5.1457 |
| OOS R² | -0.4288 |
| Directional Accuracy | 51.76% |

## Trade Statistics

- **Total Trades**: 207
- **Buy Orders**: 118 (57.0%)
- **Sell Orders**: 55 (26.6%)
- **Close Positions**: 34 (16.4%)
- **Average per Stock**: 20.7

## Results by Stock

| Stock | Total Return | Excess Return | Sharpe | Max DD | DA | OOS R² | Trades | ε |
|-------|--------------|---------------|--------|--------|-------|--------|--------|----|
| CARR | 7.15% | 19.44% | 0.20 | 15.77% | 48.14% | -0.282 | 16 | 0.0045 |
| OTIS | -31.80% | -26.92% | -3.59 | 33.76% | 55.05% | -0.084 | 28 | 0.0025 |
| RSG | 14.50% | 1.39% | 2.45 | 10.62% | 55.85% | -0.244 | 27 | 0.0020 |
| CTAS | 25.73% | 14.71% | 5.25 | 15.96% | 52.93% | -0.252 | 19 | 0.0024 |
| AJG | 14.40% | 12.10% | 1.08 | 10.68% | 53.72% | -0.524 | 28 | 0.0028 |
| PEP | -1.57% | 4.91% | -4.58 | 12.21% | 48.94% | -0.477 | 23 | 0.0026 |
| TT | 9.37% | -13.96% | 0.34 | 15.03% | 52.93% | -0.089 | 16 | 0.0032 |
| GWW | 16.82% | 1.42% | 1.39 | 11.81% | 48.40% | -1.128 | 22 | 0.0040 |
| RMD | -2.42% | -34.20% | -1.73 | 8.13% | 52.39% | -0.138 | 11 | 0.0047 |
| BLK | 13.47% | -29.39% | 1.24 | 5.50% | 49.20% | -1.069 | 17 | 0.0036 |

## Best Performers

### Top 3 by Total Return

1. **CTAS**: 25.73% (Excess: 14.71%)
2. **GWW**: 16.82% (Excess: 1.42%)
3. **RSG**: 14.50% (Excess: 1.39%)

### Top 3 by Sharpe Ratio

1. **CTAS**: Sharpe 5.25 (Return: 25.73%)
2. **RSG**: Sharpe 2.45 (Return: 14.50%)
3. **GWW**: Sharpe 1.39 (Return: 16.82%)

### Top 3 by Directional Accuracy

1. **RSG**: 55.85% (Return: 14.50%)
2. **OTIS**: 55.05% (Return: -31.80%)
3. **AJG**: 53.72% (Return: 14.40%)

## Hyperparameters by Stock

| Stock | C | epsilon | gamma | lookback_window | prediction_horizon |
|-------|-------|-------|-------|-------|-------|
| CARR | 0.1 | 0.1 | 0.001 | 60 | 5 |
| OTIS | 1 | 0.01 | 10 | 60 | 5 |
| RSG | 0.1 | 0.001 | 0.001 | 60 | 5 |
| CTAS | 0.1 | 0.001 | 0.001 | 60 | 5 |
| AJG | 0.1 | 0.01 | 0.01 | 60 | 5 |
| PEP | 1000 | 0.01 | 0.001 | 60 | 5 |
| TT | 0.1 | 0.001 | 10 | 60 | 5 |
| GWW | 0.1 | 0.01 | 1 | 60 | 5 |
| RMD | 0.1 | 0.05 | 0.01 | 60 | 5 |
| BLK | 10 | 0.05 | 0.01 | 60 | 5 |
