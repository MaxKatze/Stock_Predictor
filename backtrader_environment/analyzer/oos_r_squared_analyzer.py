from .general_analyzer import GeneralAnalyzer
import math


class OOSRSquaredAnalyzer(GeneralAnalyzer):
    """Out-of-Sample R² Analyzer (vs. Random Walk baseline).

    OOS_R² = 1 - sum((actual - predicted)²) / sum((actual - rw_prediction)²)

    Random walk prediction: previous price (i.e., predicted return = 0).
    OOS_R² > 0 means model beats random walk.
    """

    def __init__(self):
        super().__init__()
        self.previous_prediction = {}
        self.previous_price = {}
        self.ss_model = {}
        self.ss_rw = {}
        self.oos_r2 = {}
        self.num_points = {}
        for d in self.strategy.datas:
            self.previous_prediction[d] = None
            self.previous_price[d] = None
            self.ss_model[d] = 0.0
            self.ss_rw[d] = 0.0
            self.oos_r2[d._name] = 0.0
            self.num_points[d] = 0

    def next(self):
        for d in self.strategy.datas:
            prev_pred = self.previous_prediction[d]
            prev_price = self.previous_price[d]
            actual = self.strategy.current_price[d]

            if prev_pred is not None and not math.isnan(prev_pred) and prev_price is not None:
                self.ss_model[d] += (actual - prev_pred) ** 2
                self.ss_rw[d] += (actual - prev_price) ** 2
                self.num_points[d] += 1

            self.previous_prediction[d] = self.strategy.current_prediction[d]
            self.previous_price[d] = actual

    def stop(self):
        for d in self.strategy.datas:
            if self.num_points[d] < 3 or self.ss_rw[d] == 0:
                self.oos_r2[d._name] = -1
            else:
                self.oos_r2[d._name] = 1 - (self.ss_model[d] / self.ss_rw[d])

    def get_analysis(self):
        super_analysis = super().get_analysis()
        return {"general": super_analysis, "oos_r_squared": self.oos_r2}
