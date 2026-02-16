import backtrader as bt

class GeneralAnalyzer(bt.Analyzer):
    def __init__(self):
        if self.__class__ not in getattr(self.strategy, "supported_analyzers", []):
            raise ValueError(f"Analyzer not supported for Strategy: {self.__class__}")