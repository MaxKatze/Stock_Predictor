from .general_analyzer import GeneralAnalyzer


class RSquaredAnalyzer(GeneralAnalyzer):
    def __init__(self):
        super().__init__()
        self.r2 = {}
        for d in self.strategy.datas:
            self.r2[d._name] = 0

    def stop(self):
        for d in self.strategy.datas:
            if len(self.strategy.prices[d]) != len(self.strategy.predictions[d]):
                raise ValueError("Price-Prediction mismatch.")

            pairs = [
                (price, prediction)
                for price, prediction in zip(self.strategy.prices[d], self.strategy.predictions[d])
                if prediction is not None
            ]
            if len(pairs) < 3:
                self.r2[d._name] = -1
                continue

            observed_prices = [price for price, _ in pairs]
            mean_price = sum(observed_prices) / len(observed_prices)
            ss_tot = sum((price - mean_price) ** 2 for price, _ in pairs)
            if ss_tot == 0:
                self.r2[d._name] = -1
                continue

            ss_res = sum((price - prediction) ** 2 for price, prediction in pairs)
            self.r2[d._name] = 1 - (ss_res / ss_tot)

    def get_analysis(self):
        super_analysis = super().get_analysis()
        analysis = {}
        analysis["general"] = super_analysis
        analysis["r_squared_analysis"] = self.r2
        return analysis
