from xgboost import XGBClassifier
import numpy as np
from .base import BaseModel

class XGBModel(BaseModel):
    def __init__(self, **kw):
        self.clf = XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.07,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            random_state=42, eval_metric="logloss", **kw
        )
    def fit(self, X, y):
        self.clf.fit(X, y); return self
    def predict(self, X) -> np.ndarray:
        return self.clf.predict(X)
    def predict_proba(self, X):
        return self.clf.predict_proba(X)
