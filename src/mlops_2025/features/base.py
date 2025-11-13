from abc import ABC, abstractmethod
from typing import Tuple, List
import numpy as np
import pandas as pd

class FeatureEngineer(ABC):
    @abstractmethod
    def build_pipeline(self): ...
    @abstractmethod
    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]: ...
