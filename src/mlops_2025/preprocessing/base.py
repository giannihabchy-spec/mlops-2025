# src/mlops_2025/preprocessing/base.py
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple
import pandas as pd

class Preprocessor(ABC):
    @abstractmethod
    def load(self, train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]: ...
    @abstractmethod
    def clean(self, train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame: ...
    @abstractmethod
    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]: ...

    def save(self, train: pd.DataFrame, test: pd.DataFrame, out_train: str, out_test: str) -> None:
        Path(out_train).parent.mkdir(parents=True, exist_ok=True)
        Path(out_test).parent.mkdir(parents=True, exist_ok=True)
        train.to_csv(out_train, index=False)
        test.to_csv(out_test, index=False)

    def preprocess(self, train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train, test = self.load(train_path, test_path)
        df = self.clean(train, test)
        return self.split(df)
