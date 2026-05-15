# LinearSeparation Trade Statistics Summary

Evaluation Date: 2026-05-15

## Overview

Dieses Dokument zeigt die tatsächliche Handelsaktivität des LinearSeparation-Modells während der Walk-Forward-Evaluation auf dem Testdatensatz (Juli 2024 - Dezember 2025, 377 Tage).

## Trade Statistics by Stock

| Stock | Buy Orders | Sell Orders | Close Positions | Total Trades | Total Return | Directional Accuracy |
|-------|------------|-------------|-----------------|--------------|--------------|---------------------|
| CARR  | 5          | 4           | 0               | 9            | -7.00%       | 0.468               |
| OTIS  | 7          | 3           | 4               | 14           | -21.02%      | 0.479               |
| RSG   | 9          | 5           | 1               | 15           | 19.93%       | 0.561               |
| CTAS  | 7          | 6           | 0               | 13           | -3.46%       | 0.551               |
| AJG   | 9          | 4           | 3               | 16           | 19.00%       | 0.535               |
| PEP   | 5          | 0           | 3               | 8            | -13.00%      | 0.479               |
| TT    | 8          | 5           | 1               | 14           | 4.49%        | 0.497               |
| GWW   | 5          | 4           | 0               | 9            | 0.50%        | 0.481               |
| RMD   | 6          | 3           | 2               | 11           | 8.87%        | 0.535               |
| BLK   | 7          | 5           | 1               | 13           | 32.28%       | 0.548               |

## Summary Statistics

- **Average Trades per Stock**: 12.2
- **Total Trades across all Stocks**: 122
- **Average Buy Orders**: 6.8
- **Average Sell Orders**: 3.9
- **Average Close Positions**: 1.5

## Trade Frequency Analysis

- **Test Period**: 377 trading days
- **Average Trade Frequency**: 1 trade every ~31 days per stock
- **Range**: 8 trades (PEP) to 16 trades (AJG)

## Key Observations

1. **Moderate Trading Activity**: Im Durchschnitt werden etwa 12 Trades pro Aktie über den gesamten Testzeitraum ausgeführt. Dies deutet auf eine eher konservative Strategie hin, die nicht zu häufig handelt.

2. **Buy vs Sell Balance**: 
   - Buy Orders (68 total) überwiegen Sell Orders (39 total)
   - Dies zeigt eine Tendenz zu Long-Positionen
   - Close Positions (15 total) werden verwendet, um bei negativen Signalen aus dem Markt zu gehen

3. **Trade Distribution**:
   - Minimum: 8 Trades (PEP)
   - Maximum: 16 Trades (AJG)
   - Die Varianz ist relativ gering, was auf konsistente Signalgenerierung hindeutet

4. **Correlation with Performance**:
   - Hohe Trade-Anzahl korreliert nicht direkt mit besserer Performance
   - Bester Performer (BLK: +32.28%) hat 13 Trades
   - Schlechtester Performer (OTIS: -21.02%) hat 14 Trades

## Signal Cooldown Impact

Die Strategie verwendet folgende Parameter:
- `signal_cooldown = 5`: Mindestens 5 Tage zwischen Trades
- `forced_signal_interval = 20`: Spätestens alle 20 Tage wird ein Signal erzwungen

Dies begrenzt die maximale Handelsfrequenz auf etwa:
- **Theoretisches Maximum**: ~75 Trades (377 Tage / 5 Tage)
- **Tatsächlich beobachtet**: 8-16 Trades
- **Ausnutzungsgrad**: 11-21% des theoretischen Maximums

Die niedrige Ausnutzung deutet darauf hin, dass:
1. Das Modell oft Signale im Hold-Bereich (-ε ≤ r̂ ≤ ε) generiert
2. Der `forced_signal_interval` nur selten greift
3. Die Strategie selektiv handelt, basierend auf klaren Signalen

## Recommendation

Die Trade-Frequenz erscheint angemessen für eine Vorhersage-basierte Strategie. Eine zu hohe Frequenz würde auf Overfitting hindeuten, während eine zu niedrige Frequenz die Strategie ineffektiv machen würde. Mit ~12 Trades über 377 Tage liegt die Strategie in einem vernünftigen Bereich.
