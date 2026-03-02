from .general_analyzer import GeneralAnalyzer
import math

#Error Analyzer, works on return prediction
# => compares prediction to mean return => (near 1: excellent prediciton; 0: close to mean, negative: worse than taking just mean)
class RSquaredAnalyzer(GeneralAnalyzer):
    def __init__(self):
        super().__init__()
        self.price_sum = {}
        self.r2 = {}
        for d in self.strategy.datas:
            self.price_sum[d] = 0
            self.r2[d._name] = 0
        
        
    def next(self):
        for d in self.strategy.datas:
            self.price_sum[d] += self.strategy.current_price[d]


    def stop(self):
        for d in self.strategy.datas:
            mean_price = self.price_sum[d] / len(self.strategy.current_price[d])
            ss_tot = 0
            ss_res = 0
            if len(self.strategy.prices[d]) != len(self.strategy.predictions[d]):
                raise ValueError("Price-Prediction mismatch.")
            for (price, prediction) in zip(self.strategy.prices[d], self.strategy.predictions[d]):
                if prediction is not None:
                    ss_tot += (price - mean_price) * (price - mean_price)
                    ss_res += (price - prediction) * (price - prediction)
            R2 = 1 - (ss_res / ss_tot)
            self.r2[d._name] = R2

            

    def get_analysis(self):
        super_analysis = super().get_analysis()
        analysis = {}
        analysis["general"] = super_analysis
        analysis["r_squared_analysis"] = self.r2
        return analysis