from .general_analyzer import GeneralAnalyzer
import math


class MAPEAnalyzer(GeneralAnalyzer):
    """Mean Absolute Percentage Error Analyzer.

    MAPE = (1/n) * sum(|actual - predicted| / |actual|)
    """

    def __init__(self):
        super().__init__()
        self.previous_prediction = {}
        self.error_sum = {}
        self.mape = {}
        self.num_points = {}
        for d in self.strategy.datas:
            self.previous_prediction[d] = None
            self.error_sum[d] = 0.0
            self.mape[d._name] = 0.0
            self.num_points[d] = 0

    def next(self):
        for d in self.strategy.datas:
            prev_pred = self.previous_prediction[d]
            if prev_pred is not None and not math.isnan(prev_pred):
                actual = self.strategy.current_price[d]
                if actual != 0:
                    self.error_sum[d] += abs(actual - prev_pred) / abs(actual)
                    self.num_points[d] += 1
            self.previous_prediction[d] = self.strategy.current_prediction[d]

    def stop(self):
        for d in self.strategy.datas:
            if self.num_points[d] < 3:
                self.mape[d._name] = -1
            else:
                self.mape[d._name] = self.error_sum[d] / self.num_points[d]

    def get_analysis(self):
        super_analysis = super().get_analysis()
        return {"general": super_analysis, "mape": self.mape}
