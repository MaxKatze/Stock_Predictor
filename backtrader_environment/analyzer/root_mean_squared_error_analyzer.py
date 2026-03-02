from .general_analyzer import GeneralAnalyzer
import math

#Error Analyzer, works on price prediction and return prediction
class RootMeanSquaredErrorAnalyzer(GeneralAnalyzer):
    def __init__(self):
        super().__init__()
        self.previous_prediction, self.current_error_sum, self.mean_squared_error, self.num_of_data_points = {}, {}, {}, {}
        for d in self.strategy.datas:
            self.previous_prediction[d] = None
            self.current_error_sum[d] = 0
            self.mean_squared_error[d._name] = 0
            self.num_of_data_points[d] = 0
    
    def next(self):
        for d in self.strategy.datas:
            if self.previous_prediction[d] is not None:
                self.current_error_sum[d] += (self.previous_prediction[d] - self.strategy.current_price[d]) * (self.previous_prediction[d] - self.strategy.current_price[d])
                self.num_of_data_points[d] += 1
            self.previous_prediction[d] = self.strategy.current_prediction[d]


    def stop(self):
        for d in self.strategy.datas:
            if self.num_of_data_points[d] < 3:
                self.mean_squared_error[d._name] = -1
            else:
                self.mean_squared_error[d._name] = math.sqrt(self.current_error_sum[d] / self.num_of_data_points[d])

    def get_analysis(self):
        super_analysis = super().get_analysis()
        analysis = {}
        analysis["general"] = super_analysis
        analysis["root_mean_squared_error"] = self.mean_squared_error
        return analysis