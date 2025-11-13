import argparse, pandas as pd
from mlops_2025.features import TitanicFeaturizer
from mlops_2025.models import LogisticModel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--target", default="Survived")
    ap.add_argument("--model_out", required=True)
    ap.add_argument("--featurizer_out", default="./data/featurizer.joblib")
    args = ap.parse_args()

    df = pd.read_csv(args.train_csv)
    y = df[args.target].astype(int).values
    X, _ = TitanicFeaturizer().fit(df.drop(columns=[args.target])).transform(df.drop(columns=[args.target]))

    model = LogisticModel().fit(X, y)
    model.save(args.model_out)
    TitanicFeaturizer().fit(df.drop(columns=[args.target])).save(args.featurizer_out)  # or reuse the same instance you fitted

    print(f"✅ Model -> {args.model_out}")
    print(f"✅ Featurizer -> {args.featurizer_out}")

if __name__ == "__main__":
    main()
