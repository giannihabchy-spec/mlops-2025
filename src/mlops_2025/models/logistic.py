from sklearn.linear_model import LogisticRegression
import numpy as np
from .base import BaseModel

class LogisticModel(BaseModel):
    def __init__(self, **kw):
        self.clf = LogisticRegression(max_iter=1000, **kw)

    def fit(self, X, y):
        self.clf.fit(X, y); return self

    def predict(self, X) -> np.ndarray:
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)
