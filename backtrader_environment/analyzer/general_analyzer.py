import backtrader as bt

class GeneralAnalyzer(bt.Analyzer):
    def __init__(self):
        if self.__class__ not in getattr(self.strategy, "supported_analyzers", []):
            raise ValueError(f"""Analyzer not supported for Strategy: {self.__class__}. If you think this might be an error, 
                             check 'supported_analyzers' property of the registered strategies.""")


# Backtrader – Built-in Analyzer

# Performance / Returns
# - AnnualReturn
# - Returns
# - TimeReturn
# - LogReturnsRolling
# - PeriodStats
# - VWR (Variability-Weighted Return)

# Risiko / Kennzahlen
# - SharpeRatio
# - SharpeRatio_A
# - Calmar
# - SQN (System Quality Number)

# Drawdown
# - DrawDown
# - TimeDrawDown

# Trades / Portfolio
# - TradeAnalyzer
# - Transactions
# - GrossLeverage
# - PositionsValue

# Integration
# - PyFolio