# Stock_Predictor

Stock Prediction

Python 3.13.7

Run:

just use uv or ask cursor how to run.

´´´bash
uv run main.py

Repo Structure:

Entry Point: main.py

Registrates Strategy, Analyzer and Models

Flow: Register Strategy in main. Strategy instantiates a Model, that is fitted and predicts.
Use Analyzer to analyze trades and predictions, e.g. Lsq
Use Sizers to determine sizing of trades. (sizer gets called when size attribute in trade call is not used)

