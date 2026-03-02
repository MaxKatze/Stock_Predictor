from .general_analyzer import GeneralAnalyzer
import math


class MeanAbsoluteErrorAnalyzer(GeneralAnalyzer):
    def __init__(self):
        super().__init__()
        self.previous_prediction, self.current_error_sum, self.mean_absolute_error, self.num_of_data_points = {}, {}, {}, {}
        for d in self.strategy.datas:
            self.previous_prediction[d] = None
            self.current_error_sum[d] = 0
            self.mean_absolute_error[d._name] = 0
            self.num_of_data_points[d] = 0
    
    def next(self):
        for d in self.strategy.datas:
            prev_prediction = self.previous_prediction[d]
            if prev_prediction is not None and not math.isnan(prev_prediction):
                self.current_error_sum[d] += abs(prev_prediction - self.strategy.current_price[d])
                self.num_of_data_points[d] += 1
            self.previous_prediction[d] = self.strategy.current_prediction[d]


    def stop(self):
        for d in self.strategy.datas:
            if self.num_of_data_points[d] < 3:
                self.mean_absolute_error[d._name] = -1
            else:
                self.mean_absolute_error[d._name] = self.current_error_sum[d] / self.num_of_data_points[d]

    def get_analysis(self):
        super_analysis = super().get_analysis()
        analysis = {}
        analysis["general"] = super_analysis
        analysis["mean_absolute_error"] = self.mean_absolute_error
        return analysis
