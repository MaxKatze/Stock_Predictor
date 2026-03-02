from abc import ABC, abstractmethod


class PredictionModel(ABC):
    @abstractmethod
    def predict(self, n):
        raise NotImplementedError

    @abstractmethod
    def fit(self, data, window):
        raise NotImplementedError
