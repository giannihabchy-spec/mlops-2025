# src/mlops_2025/features/titanic.py
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from .base import FeatureEngineer


class TitanicFeaturizer(FeatureEngineer):
    """
    Stateful featurizer: fit once on training data, then reuse for eval/inference.
    """

    def __init__(self) -> None:
        self.pipe: Optional[Pipeline] = None

    # -------------------------
    # Helpers
    # -------------------------
    @staticmethod
    def extract_title(df: pd.DataFrame) -> pd.DataFrame:
        t = df["Name"].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
        t = t.replace(
            ["Lady", "the Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev", "Sir", "Jonkheer", "Dona"],
            "Rare",
        )
        t = t.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
        df["Title"] = t
        return df

    @staticmethod
    def compute_family_size(df: pd.DataFrame) -> pd.DataFrame:
        df["Family_size"] = df.get("SibSp", 0) + df.get("Parch", 0) + 1

        def family_size(n: int) -> str:
            if n == 1:
                return "Alone"
            elif 1 < n < 5:
                return "Small"
            else:
                return "Large"

        df["Family_size"] = df["Family_size"].apply(family_size)
        return df

    @staticmethod
    def tidy_drop(df: pd.DataFrame) -> pd.DataFrame:
        # Minimal imputations to avoid NaNs at inference
        if "Age" in df:
            df["Age"] = df["Age"].fillna(df["Age"].median()).astype("int64")
        if "Fare" in df:
            df["Fare"] = df["Fare"].fillna(df["Fare"].median())
        if "Embarked" in df and df["Embarked"].isna().any():
            df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode().iloc[0])

        # Drop raw text / leak-prone columns
        return df.drop(columns=["Name", "Parch", "SibSp", "Ticket", "Cabin"], errors="ignore")

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.extract_title(df.copy())
        df = self.compute_family_size(df)
        df = self.tidy_drop(df)

        # Enforce a stable column order expected by the transformers
        want_order = ["Age", "Sex", "Fare", "Embarked", "Pclass", "Title", "Family_size"]
        existing = [c for c in want_order if c in df.columns]
        ordered = pd.concat([df[existing], df.drop(columns=existing)], axis=1)
        return ordered

    # -------------------------
    # Pipeline
    # -------------------------
    def build_pipeline(self) -> Pipeline:
        """
        Indices assume the ordered columns:
        0: Age (num), 1: Sex (cat), 2: Fare (num), 3: Embarked (cat),
        4: Pclass (ord), 5: Title (cat), 6: Family_size (cat)
        """
        num_cat = ColumnTransformer(
            transformers=[
                ("scaling", MinMaxScaler(), [0, 2]),  # Age, Fare
                ("onehot_1", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), [1, 3]),  # Sex, Embarked
                ("ordinal", OrdinalEncoder(), [4]),  # Pclass
                ("onehot_2", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), [5, 6]),  # Title, Family_size
            ],
            remainder="drop",  # keep it stable across train/infer
        )

        # Binning (optional extra features) on Age/Fare positions *after* previous step.
        # Note: This uses positional columns of the previous output; it worked in your runs.
        bins = ColumnTransformer(
            transformers=[
                ("Kbins", KBinsDiscretizer(n_bins=15, encode="ordinal", strategy="quantile"), [0, 2]),
            ],
            remainder="passthrough",
        )

        return Pipeline(
            steps=[
                ("num_cat_transformation", num_cat),
                ("bins", bins),
            ]
        )

    # -------------------------
    # Public API
    # -------------------------
    def fit(self, df: pd.DataFrame) -> "TitanicFeaturizer":
        ordered = self._prepare(df)
        self.pipe = self.build_pipeline().fit(ordered)
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        if self.pipe is None:
            raise RuntimeError("Featurizer not fitted. Call fit() first.")
        ordered = self._prepare(df)
        X = self.pipe.transform(ordered)
        try:
            names = self.pipe.get_feature_names_out().tolist()
        except Exception:
            names = [f"x{i}" for i in range(X.shape[1])]
        return X, names

    # Persistence
    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "TitanicFeaturizer":
        return joblib.load(path)
