import argparse
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def extract_title(df: pd.DataFrame) -> pd.DataFrame:
    t = df['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
    t = t.replace(['Lady','the Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'], 'Rare')
    t = t.replace({'Mlle':'Miss','Ms':'Miss','Mme':'Mrs'})
    df['Title'] = t
    return df

def compute_family_size(df: pd.DataFrame) -> pd.DataFrame:
    df['Family_size'] = df['SibSp'] + df['Parch'] + 1
    def family_size(number):
        if number == 1: return "Alone"
        elif 1 < number < 5: return "Small"
        else: return "Large"
    df['Family_size'] = df['Family_size'].apply(family_size)
    return df

def tidy_drop(df: pd.DataFrame) -> pd.DataFrame:
    if 'Age' in df:
        df['Age'] = df['Age'].fillna(df['Age'].median()).astype('int64')
    return df.drop(columns=['Name','Parch','SibSp','Ticket'], errors='ignore')

def build_feature_pipeline():
    num_cat_transformation = ColumnTransformer([
        ('scaling', MinMaxScaler(), [0, 2]),
        ('onehot_1', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), [1, 3]),
        ('ordinal', OrdinalEncoder(), [4]),
        ('onehot_2', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), [5, 6]),
    ], remainder='passthrough')

    bins = ColumnTransformer([
        ('Kbins', KBinsDiscretizer(n_bins=15, encode='ordinal', strategy='quantile'), [0, 2]),
    ], remainder='passthrough')

    return Pipeline([
        ('num_cat_transformation', num_cat_transformation),
        ('bins', bins),
    ])

def featurize(df: pd.DataFrame) -> (np.ndarray, list):
    df = extract_title(df.copy())
    df = compute_family_size(df)
    df = tidy_drop(df)

    want_order = ['Age','Sex','Fare','Embarked','Pclass','Title','Family_size']
    existing = [c for c in want_order if c in df.columns]
    ordered_df = pd.concat([df[existing], df.drop(columns=existing)], axis=1)

    pipe = build_feature_pipeline()
    X = pipe.fit_transform(ordered_df)

    try:
        names = pipe.get_feature_names_out()
    except Exception:
        names = [f'x{i}' for i in range(X.shape[1])]

    return X, names

def main():
    ap = argparse.ArgumentParser(description="Featurize Titanic-like data into a model-ready matrix.")
    ap.add_argument("--input", "-i", required=True, help="Input CSV")
    ap.add_argument("--output", "-o", required=True, help="Output CSV for features")
    ap.add_argument("--target", "-y", default=None, help="Optional target column to append to output")
    args = ap.parse_args()

    df = pd.read_csv(args.input)

    y = None
    if args.target and args.target in df.columns:
        y = df[args.target].copy()
        df = df.drop(columns=[args.target])

    X, names = featurize(df)

    out = pd.DataFrame(X, columns=names)
    if y is not None:
        out[args.target] = y.values

    out.to_csv(args.output, index=False)

if __name__ == "__main__":
    main()
