from .general_analyzer import GeneralAnalyzer

class LeastSquareAnalyzer(GeneralAnalyzer):
    def __init__(self):
        super().__init__()
    
    def next(self):
        pass

    def stop(self):
        pass

    def get_analysis(self):
        return super().get_analysis()
