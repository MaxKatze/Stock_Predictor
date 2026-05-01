from .general_analyzer import GeneralAnalyzer
import math


class DirectionalAccuracyAnalyzer(GeneralAnalyzer):
    """Directional Accuracy Analyzer.

    DA = (1/n) * sum(1 if sign(actual_move) == sign(predicted_move))

    Where:
      actual_move = price_t - price_{t-1}
      predicted_move = prediction_t - price_{t-1}

    DA = 1.0: perfect direction prediction
    DA = 0.5: random guessing
    """

    def __init__(self):
        super().__init__()
        self.previous_prediction = {}
        self.previous_price = {}
        self.correct_directions = {}
        self.da = {}
        self.num_points = {}
        for d in self.strategy.datas:
            self.previous_prediction[d] = None
            self.previous_price[d] = None
            self.correct_directions[d] = 0
            self.da[d._name] = 0.0
            self.num_points[d] = 0

    def next(self):
        for d in self.strategy.datas:
            prev_pred = self.previous_prediction[d]
            prev_price = self.previous_price[d]
            actual = self.strategy.current_price[d]

            if prev_pred is not None and not math.isnan(prev_pred) and prev_price is not None:
                actual_move = actual - prev_price
                predicted_move = prev_pred - prev_price

                if _sign(actual_move) == _sign(predicted_move):
                    self.correct_directions[d] += 1
                self.num_points[d] += 1

            self.previous_prediction[d] = self.strategy.current_prediction[d]
            self.previous_price[d] = actual

    def stop(self):
        for d in self.strategy.datas:
            if self.num_points[d] < 3:
                self.da[d._name] = -1
            else:
                self.da[d._name] = self.correct_directions[d] / self.num_points[d]

    def get_analysis(self):
        super_analysis = super().get_analysis()
        return {"general": super_analysis, "directional_accuracy": self.da}


def _sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    return 0
