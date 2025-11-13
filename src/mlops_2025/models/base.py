from abc import ABC, abstractmethod
import joblib
import numpy as np

class BaseModel(ABC):
    @abstractmethod
    def fit(self, X, y): ...
    @abstractmethod
    def predict(self, X) -> np.ndarray: ...
    def predict_proba(self, X): return None

    def save(self, path: str): joblib.dump(self, path)
    @classmethod
    def load(cls, path: str): return joblib.load(path)
