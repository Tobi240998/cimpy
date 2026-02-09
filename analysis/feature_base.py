from abc import ABC, abstractmethod


class BaseFeatureExtractor(ABC):
    def __init__(self):
        self.features = {}

    @abstractmethod
    def extract(self) -> dict:
        pass

    def get_features(self) -> dict:
        if not self.features:
            self.features = self.extract()
        return self.features
