from sklearn.ensemble import RandomForestClassifier
import numpy as np
from .base import BaseModel

class RandomForestModel(BaseModel):
    def __init__(self, **kw):
        self.clf = RandomForestClassifier(n_estimators=300, random_state=42, **kw)
    def fit(self, X, y):
        self.clf.fit(X, y); return self
    def predict(self, X) -> np.ndarray:
        return self.clf.predict(X)
    def predict_proba(self, X):
        return self.clf.predict_proba(X)
