# LinearSeparation Trade Statistics - Evaluation Results

**Evaluation Date**: 2026-05-15  
**Model**: LinearSeparation (ARIMA-based)  
**Test Period**: July 2024 - December 2025 (377 trading days)  
**Initial Capital**: $100,000 per stock  

## 📁 Generated Files

### Data Files
- **`evaluation_results.json`** - Complete evaluation results including all metrics and trade statistics
- **`trade_statistics.csv`** - CSV export of trade statistics for further analysis

### Documentation
- **`trade_statistics_summary.md`** - Detailed written analysis of trade statistics

### Visualizations
- **`trade_statistics_visualization.png`** - 6-panel comprehensive trade analysis dashboard
- **`trade_statistics_summary_charts.png`** - 4-panel summary charts with efficiency analysis
- **`walkforward_*.png`** - Individual walk-forward plots for each stock (10 files)

### Analysis Scripts
- **`analyze_trades.py`** - Extract and analyze trade statistics from JSON
- **`visualize_trades.py`** - Create comprehensive visualizations

## 📊 Key Findings

### Trade Activity Summary

| Metric | Value |
|--------|-------|
| Total Trades (all stocks) | **122** |
| Average Trades per Stock | **12.2** |
| Total Buy Orders | **68** (55.7%) |
| Total Sell Orders | **39** (32.0%) |
| Total Close Positions | **15** (12.3%) |

### Trade Frequency

- **Average days between trades**: 30.9 days
- **Theoretical maximum trades**: 75 (with 5-day cooldown)
- **Actual utilization**: 16.2%

👉 **Interpretation**: Die Strategie handelt selektiv und nutzt nur etwa 16% der theoretisch möglichen Trades. Dies deutet darauf hin, dass:
1. Die meisten Prognosen im Hold-Bereich (-ε ≤ r̂ ≤ ε) liegen
2. Das Modell konservativ agiert und nur bei klaren Signalen handelt
3. Der `forced_signal_interval` (20 Tage) selten greift

### Performance Distribution

| Category | Stocks | Return Range |
|----------|--------|--------------|
| **Profitable** | 5 | +0.5% to +32.3% |
| **Loss-making** | 5 | -21.0% to -3.5% |

**Top 3 Performers**:
1. **BLK**: +32.28% (13 trades)
2. **RSG**: +19.93% (15 trades)
3. **AJG**: +19.00% (16 trades)

**Bottom 3 Performers**:
1. **OTIS**: -21.02% (14 trades)
2. **PEP**: -13.00% (8 trades)
3. **CARR**: -7.00% (9 trades)

### Correlation Analysis

| Correlation | Value | Interpretation |
|-------------|-------|----------------|
| Trades ↔ Return | **+0.455** | Moderate positive: Mehr Trades tendieren zu besserer Performance |
| Trades ↔ Directional Accuracy | **+0.613** | Strong positive: Mehr Trades korrelieren mit besserer Richtungsgenauigkeit |
| Trades ↔ OOS-R² | **-0.340** | Negative: Mehr Trades korrelieren mit schlechterer Out-of-Sample R² |

**Interessanter Befund**: Die positive Korrelation zwischen Trades und Return (+0.455) widerspricht der intuitiven Erwartung, dass zu viele Trades zu Overtrading führen. Dies deutet darauf hin, dass:
- Die Strategie bei aktiv handelbaren Aktien (mit klaren Trends) bessere Performance erzielt
- Aktien mit wenigen Trades oft schwierige Marktbedingungen aufweisen (seitwärts, volatil)

### Trade Type Analysis

**Buy vs. Sell Imbalance**:
- Buy Orders (68) überwiegen Sell Orders (39) um 74%
- Close Positions (15) werden genutzt, um bei negativen Signalen zu verkaufen

**Implikationen**:
1. Die Strategie ist **Long-biased** (bevorzugt Long-Positionen)
2. Half-Kelly Position Sizing führt zu häufigeren Rebalancing-Käufen als Verkäufen
3. Close Positions werden strategisch bei starken Verkaufssignalen eingesetzt

## 🎯 Strategische Parameter

Die Evaluation verwendet folgende Strategie-Parameter:

```python
signal_threshold = MAE * 0.3  # Schwellenwert ε basierend auf Validierungsdaten
signal_cooldown = 5           # Min. 5 Tage zwischen Trades
forced_signal_interval = 20   # Spätestens alle 20 Tage ein Signal
kelly_window = 60             # Fenster für Half-Kelly Berechnung
```

**Optimierungspotential**:
- `signal_cooldown` könnte reduziert werden (z.B. 3 Tage), um reaktiver zu sein
- `forced_signal_interval` könnte entfernt werden, um nur bei klaren Signalen zu handeln
- Signal Threshold könnte dynamisch angepasst werden (volatilitätsbasiert)

## 📈 Model Performance Metrics

**Average across all stocks**:
- Total Return: **4.06%**
- Excess Return vs. Buy-and-Hold: **-7.56%**
- Sharpe Ratio: **-2.83** (negativ aufgrund hoher Volatilität)
- Max Drawdown: **17.58%**
- Directional Accuracy: **51.33%** (leicht besser als Random Walk)
- OOS-R²: **-0.0994** (schlechter als Random Walk Baseline)

**Interpretation**:
- Das Modell schlägt Buy-and-Hold im Durchschnitt **nicht**
- Directional Accuracy von 51.3% ist nur marginal besser als Zufall (50%)
- Negative OOS-R² zeigt, dass ARIMA die Zeitreihen nicht besser vorhersagt als ein Random Walk
- Trotzdem gibt es einzelne Aktien (BLK, RSG, AJG) mit starker positiver Performance

## 🔍 Nächste Schritte

1. **Model Comparison**: Vergleich mit SVR und TimesFM Modellen
2. **Parameter Optimization**: Grid-Search für `signal_cooldown`, `forced_signal_interval`
3. **Feature Engineering**: Integration von Makrofaktoren oder Sentiment-Daten
4. **Ensemble Approach**: Kombination mehrerer Modelle
5. **Risk Management**: Verbesserung des Position Sizing (z.B. volatilitätsbasiert)

## 📚 Usage

### Run Evaluation
```bash
cd backtrader_environment
uv run python run_evaluation.py --models LinearSeparation
```

### Analyze Trade Statistics
```bash
uv run python analyze_trades.py
```

### Create Visualizations
```bash
uv run python visualize_trades.py
```

## 🛠️ Code Changes

Die folgenden Änderungen wurden am Code vorgenommen:

1. **`strategies/prediction_strategy.py`**:
   - Hinzugefügt: `buy_count`, `sell_count`, `close_count` Tracking
   - Modifiziert: `_handle_long_signal()` und `_handle_cash_signal()` zum Zählen
   - Modifiziert: `stop()` Methode zur Ausgabe der Trade-Statistiken

2. **`run_evaluation.py`**:
   - Hinzugefügt: Trade-Statistiken in das `metrics` Dictionary
   - Modifiziert: Konsolenausgabe zur Anzeige der Trade-Anzahl

## 📝 Notes

- Die Trade-Statistiken werden **pro Datafeed** gezählt
- Bei Multiple-Stock-Strategien werden die Statistiken aggregiert
- Close Positions zählen als separate Kategorie (nicht als Sell)
- Die Strategie verwendet Market Orders (Execution zum nächsten Open)

---

*Generated by LinearSeparation Evaluation Framework*  
*Repository: Studienarbeit - Stock Predictor*
