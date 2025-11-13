# src/mlops_2025/preprocessing/titanic.py
from .base import Preprocessor
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

class TitanicPreprocessor(Preprocessor):
    def load(self, train_path, test_path):
        return pd.read_csv(train_path), pd.read_csv(test_path)
    def clean(self, train, test):
        train = train.copy(); test = test.copy()
        for df in (train, test):
            df.drop(columns=["Cabin"], inplace=True, errors="ignore")
        train["Embarked"].fillna("S", inplace=True)
        test["Fare"].fillna(test["Fare"].mean(), inplace=True)
        df = pd.concat([train, test], sort=True).reset_index(drop=True)
        df["Age"] = df.groupby(["Sex","Pclass"])["Age"].transform(lambda x: x.fillna(x.median()))
        return df
    def split(self, df):
        train, test = df.iloc[:891].copy(), df.iloc[891:].copy()
        test.drop(columns=["Survived"], inplace=True, errors="ignore")
        train["Survived"] = train["Survived"].astype(int)
        return train, test
