import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from .base import FeatureEngineer

class TitanicFeaturizer(FeatureEngineer):
    # ---- feature helpers ----
    @staticmethod
    def extract_title(df: pd.DataFrame) -> pd.DataFrame:
        t = df['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
        t = t.replace(['Lady','the Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'], 'Rare')
        t = t.replace({'Mlle':'Miss','Ms':'Miss','Mme':'Mrs'})
        df['Title'] = t
        return df

    @staticmethod
    def compute_family_size(df: pd.DataFrame) -> pd.DataFrame:
        df['Family_size'] = df['SibSp'] + df['Parch'] + 1
        def family_size(n):
            if n == 1: return "Alone"
            elif 1 < n < 5: return "Small"
            else: return "Large"
        df['Family_size'] = df['Family_size'].apply(family_size)
        return df

    @staticmethod
    def tidy_drop(df: pd.DataFrame) -> pd.DataFrame:
        if 'Age' in df:
            df['Age'] = df['Age'].fillna(df['Age'].median()).astype('int64')
        return df.drop(columns=['Name','Parch','SibSp','Ticket'], errors='ignore')

    # ---- pipeline ----
    def build_pipeline(self):
        num_cat = ColumnTransformer([
            ('scaling', MinMaxScaler(), [0, 2]),
            ('onehot_1', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), [1, 3]),
            ('ordinal', OrdinalEncoder(), [4]),
            ('onehot_2', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), [5, 6]),
        ], remainder='passthrough')

        bins = ColumnTransformer([
            ('Kbins', KBinsDiscretizer(n_bins=15, encode='ordinal', strategy='quantile'), [0, 2]),
        ], remainder='passthrough')

        return Pipeline([
            ('num_cat_transformation', num_cat),
            ('bins', bins),
        ])

    # ---- main API ----
    def transform(self, df: pd.DataFrame):
        df = self.extract_title(df.copy())
        df = self.compute_family_size(df)
        df = self.tidy_drop(df)

        want_order = ['Age','Sex','Fare','Embarked','Pclass','Title','Family_size']
        existing = [c for c in want_order if c in df.columns]
        ordered = pd.concat([df[existing], df.drop(columns=existing)], axis=1)

        pipe = self.build_pipeline()
        X = pipe.fit_transform(ordered)

        try:
            names = pipe.get_feature_names_out().tolist()
        except Exception:
            names = [f'x{i}' for i in range(X.shape[1])]

        return X, names
