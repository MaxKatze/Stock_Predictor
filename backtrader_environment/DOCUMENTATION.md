# Stock Predictor — Backtesting Framework

Dieses Projekt implementiert ein modulares Backtesting-Framework zur Vorhersage von Aktienkursen. Es kombiniert klassische Zeitreihenanalyse, maschinelles Lernen und ein Foundation Model, um Handelsstrategien auf historischen Daten zu evaluieren.

---

## Inhaltsverzeichnis

1. [Projektstruktur](#projektstruktur)
2. [Architektur](#architektur)
3. [Modelle](#modelle)
4. [Aktienauswahl (SDE)](#aktienauswahl-sde)
5. [Marktstrategie](#marktstrategie)
6. [Metriken](#metriken)
7. [Daten-Pipeline](#daten-pipeline)
8. [Installation & Ausführung](#installation--ausführung)
9. [Konfiguration](#konfiguration)

---

## Projektstruktur

```
backtrader_environment/
├── run_evaluation.py                  # Haupt-Evaluierungsskript (Walk-Forward)
├── main.py                            # Einfaches Backtest-Beispiel (Moving Average)
├── test_lstm.py                       # LSTM-spezifischer Backtest
├── config.yaml.example                # Konfigurationsvorlage
├── pyproject.toml                     # Abhängigkeiten
│
├── models/                            # Vorhersagemodelle
│   ├── prediction_models.py           # Abstrakte Basisklasse
│   ├── linear_separation_model.py     # HP-Filter + ARMA
│   ├── svr_prediction_model.py        # Support Vector Regression
│   ├── timesfm_prediction_model.py    # TimesFM Foundation Model
│   ├── arima_prediction_model.py      # ARIMA(p,d,q)
│   └── lstm_prediction_model.py       # Multi-Feature Attention LSTM
│
├── strategies/                        # Handelsstrategien
│   ├── general_strategy.py            # Basisklasse
│   ├── prediction_strategy.py         # Generische Strategie (für alle Modelle)
│   ├── moving_average_strategy.py     # MA-Crossover
│   ├── arima_strategy.py              # ARIMA-basiert
│   ├── lstm_strategy.py               # LSTM-basiert
│   └── fixed_date_strategy.py         # Feste Kauf-/Verkaufstermine
│
├── analyzer/                          # Bewertungsmetriken
│   ├── general_analyzer.py            # Basisklasse
│   ├── mape_analyzer.py               # Mean Absolute Percentage Error
│   ├── oos_r_squared_analyzer.py      # Out-of-Sample R² (vs. Random Walk)
│   ├── directional_accuracy_analyzer.py # Richtungsgenauigkeit
│   ├── mean_absolute_error_analyzer.py
│   ├── mean_squared_error_analyzer.py
│   ├── root_mean_squared_error_analyzer.py
│   └── r_squared_analyzer.py
│
├── stock_selection/                   # SDE-basierte Aktienauswahl
│   ├── sde_stock_selector.py          # Algorithmus
│   └── run_selection.py               # Ausführungsskript
│
├── data_handling/                     # Datenverwaltung
│   ├── download_assets.py             # Yahoo Finance Download
│   ├── data_loader.py                 # Laden & Splitten der Daten
│   ├── sp500_universe.py              # S&P 500 Ticker-Liste (Stand 2020)
│   └── file_format_yahoo.py           # Backtrader CSV-Feed
│
├── sizing/                            # Positionsgrößen
│   ├── GeneralSizing.py
│   └── PercentageSizing.py
│
├── visualization/                     # Plotting
│   └── ARIMALine.py
│
└── data/                              # Datenverzeichnis (wird automatisch befüllt)
    ├── assets/                        # Aktienkurse (CSV)
    ├── macro/                         # Makroökonomische Faktoren
    └── selection/                     # Ergebnis der Aktienauswahl (JSON)
```

---

## Architektur

Das Framework folgt einem modularen Aufbau mit drei Kernabstraktionen:

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ PredictionModel │────▶│ PredictionStrategy│────▶│ GeneralAnalyzer  │
│   (predict)     │     │   (buy/sell)      │     │   (bewerten)     │
└─────────────────┘     └──────────────────┘     └─────────────────┘
         │                       │                        │
    LinearSeparation       Backtrader Cerebro         MAPE, DA,
    SVR, TimesFM,          (Walk-Forward)           OOS-R², RMSE
    ARIMA, LSTM
```

**Datenfluss:**

1. Daten herunterladen (Yahoo Finance) → CSV-Cache
2. Aktienauswahl via SDE-Simulation → Top 10 Ticker
3. Daten aufteilen: 60% Training / 10% Validierung / 30% Test
4. Modell auf Trainingsdaten trainieren (Hyperparameter auf Validierung tunen)
5. Walk-Forward-Backtest auf Testdaten (Backtrader)
6. Metriken auswerten und vergleichen

---

## Modelle

### LinearSeparation

Zerlegt den Log-Preis in drei Komponenten:

```
p_t = τ_lang + τ_kurz + c_t + ε_t
```

| Komponente | Methode | Parameter |
|---|---|---|
| Langfristiger Trend (τ_lang) | HP-Filter + OLS-Extrapolation | λ=1600·252², Fenster=252 Tage |
| Kurzfristiger Trend (τ_kurz) | HP-Filter + OLS-Extrapolation | λ=1600, Fenster=40 Tage |
| Zyklische Schwankungen (c_t) | ARMA(p,q) via AIC-Grid-Search | p,q ∈ {0,...,5} |

Stationarität der Residuen wird mit dem ADF-Test geprüft.

### Preprocessed SVR

Support Vector Regression mit RBF-Kernel auf 14 technischen Features:

- Returns (1d, 5d, 10d, 20d)
- Volatilität (5d, 20d)
- RSI(14), MACD, SMA-Ratios (20, 50)
- HL-Range, OC-Change, Volume-Change

Hyperparameter-Tuning via Grid-Search auf Validierungsdaten:
- C ∈ {0.1, 1, 10, 100, 1000}
- ε ∈ {0.001, 0.01, 0.1, 0.5, 1}
- γ ∈ {0.01, 0.1, 1, 10, 50, 100}

### TimesFM

Googles TimesFM Foundation Model im Zero-Shot-Modus (kein Training auf Aktiendaten). Arbeitet auf Log-Renditen mit variablem Kontextfenster (bis 512 Tage).

Optionale Abhängigkeit — wird übersprungen wenn `timesfm` nicht installiert ist.

### ARIMA

ARIMA(2,1,1) mit konfigurierbarem Fit-Intervall und Fensterbreite.

### LSTM (Multi-Feature Attention)

Unidirektionales LSTM mit Feature-Attention und Temporal-Attention. Prognostiziert 5-Tage-Renditen. Details siehe `test_lstm.py`.

---

## Aktienauswahl (SDE)

Aus dem S&P 500 Universum (Stand 2020, ~505 Aktien) werden 10 Aktien mittels stochastischer Differentialgleichungen ausgewählt:

1. **Diskretisierung**: Tägliche Kursbewegung → {+1, 0, -1}
2. **Faktor-Regression**: Für jede Aktie wird per OLS der Einfluss von 6 Makrofaktoren bestimmt:
   - Öl (CL=F), Gold (GC=F), 10J-Treasury (^TNX)
   - S&P 500 (^GSPC), US-Dollar (DX-Y.NYB), Inflation (T10YIE)
3. **Gruppierung**: Aktien mit gleichem dominanten Faktorpaar werden in Pools gruppiert
4. **Simulation**: Pro Gruppe (4 Aktien + 2 Faktoren):
   - Wahrscheinlichkeiten mit Laplace-Glättung berechnen
   - Drift μ und Kovarianzmatrix Σ aufstellen
   - Cholesky-Zerlegung → Euler-Maruyama-Simulation
5. **Ranking**: 100 Wiederholungen, Top 10 nach durchschnittlichem Rang

---

## Marktstrategie

Die generische `PredictionStrategy` funktioniert mit jedem `PredictionModel`:

- **Kaufsignal**: Prognostizierte Rendite > Schwellenwert (default: 1%)
- **Verkaufssignal**: Prognostizierte Rendite < −Schwellenwert
- **Positionsgröße**: 95% des verfügbaren Kapitals
- **Ausführung**: Zum Open des nächsten Tages (kein Lookahead-Bias)
- **Warmup**: Mindestens 60 Bars vor erster Prognose

---

## Metriken

### Vorhersagequalität

| Metrik | Beschreibung | Gut wenn |
|---|---|---|
| RMSE | Root Mean Squared Error | niedrig |
| MAPE | Mean Absolute Percentage Error | niedrig |
| OOS-R² | Out-of-Sample R² vs. Random Walk | > 0 |
| DA | Directional Accuracy (Richtung korrekt?) | > 0.5 |

### Finanzielle Performance

| Metrik | Beschreibung |
|---|---|
| Gesamtrendite | (Endwert − Startwert) / Startwert |
| Excess Return | Strategie-Rendite − Buy-and-Hold |
| Sharpe Ratio | Rendite-Risiko-Verhältnis |
| Max Drawdown | Größter Verlust vom Peak |

---

## Daten-Pipeline

### Datensplit (60/10/30)

| Zeitraum | Verwendung | Daten |
|---|---|---|
| Training | Modelle trainieren | 01.2020 – 12.2023 |
| Validierung | Hyperparameter tunen | 01.2024 – 06.2024 |
| Test | Finale Evaluation | 07.2024 – 12.2025 |

### Bereinigung

Historische Preise werden für Aktiensplits und Dividenden adjustiert (`auto_adjust=True` in yfinance), damit Modelle auf korrekten Werten trainieren.

### Kein Lookahead-Bias

- Scaler nur auf Trainingsdaten gefittet
- Unidirektionales LSTM (kein Blick in die Zukunft)
- Nur Forward-Fill für NaN-Werte
- Kaufentscheidung basiert auf Schlusskurs → Ausführung zum nächsten Open

---

## Installation & Ausführung

### Voraussetzungen

- Python ≥ 3.13
- pip oder uv

### 1. Abhängigkeiten installieren

```bash
cd backtrader_environment
pip install -e .

# Optional: TimesFM Foundation Model
pip install timesfm
```

### 2. Konfiguration erstellen

```bash
cp config.yaml.example config.yaml
```

### 3. Daten herunterladen

```bash
# Einzelne Aktien (aus config.yaml)
python data_handling/download_assets.py

# Gesamtes S&P 500 Universum (für SDE-Auswahl)
python -c "
from data_handling.download_assets import download_universe
from data_handling.sp500_universe import get_sp500_2020_tickers
download_universe(get_sp500_2020_tickers())
"
```

### 4. Aktienauswahl durchführen (optional)

```bash
python stock_selection/run_selection.py
```

Ergebnis wird in `data/selection/selected_stocks.json` gespeichert.

### 5. Modelle evaluieren

```bash
# Alle Modelle auf allen konfigurierten Aktien
python run_evaluation.py
```

Ausgabe: Vergleichstabelle mit allen Metriken pro Modell und Aktie.

### Einzelne Strategien testen

```bash
# Moving Average Backtest
python main.py

# LSTM Backtest mit Train/Test Split
python test_lstm.py
```

---

## Konfiguration

Die `config.yaml` steuert alle Parameter:

```yaml
# Aktien die evaluiert werden
assets:
  - AAPL
  - MSFT
  - NVDA

# Datums-Splits
training_start: "2020-01-02"
training_end: "2023-12-29"
validation_start: "2024-01-02"
validation_end: "2024-06-28"
test_start: "2024-07-01"
test_end: "2025-12-31"

# Daten
force_update: false       # true = erneut herunterladen
auto_adjust: true         # Split-/Dividendenbereinigung

# Makroökonomische Faktoren (für SDE-Auswahl)
macro_factors:
  oil: "CL=F"
  gold: "GC=F"
  treasury_10y: "^TNX"
  sp500: "^GSPC"
  dxy: "DX-Y.NYB"
  inflation_10y: "T10YIE"

# SDE Aktienauswahl
sde:
  num_simulations: 100    # Wiederholungen
  group_size: 4           # Aktien pro Simulationsgruppe
  num_factors: 2          # Faktoren pro Gruppe
  num_selected: 10        # Anzahl ausgewählter Aktien
```

---

## Erweiterbarkeit

Neue Modelle können einfach hinzugefügt werden:

1. Klasse von `PredictionModel` ableiten (in `models/`)
2. `fit(data, window)` und `predict(n)` implementieren
3. In `run_evaluation.py` unter `MODEL_CONFIGS` registrieren

Neue Metriken:

1. Klasse von `GeneralAnalyzer` ableiten (in `analyzer/`)
2. `next()` und `stop()` implementieren
3. In `PredictionStrategy.supported_analyzers` eintragen
