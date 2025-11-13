import argparse, pandas as pd
from mlops_2025.features.titanic import TitanicFeaturizer
from mlops_2025.models import LogisticModel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--featurizer_path", default="./data/featurizer.joblib")
    ap.add_argument("--output_csv", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)
    fe = TitanicFeaturizer.load(args.featurizer_path)
    X, _ = fe.transform(df)

    m = LogisticModel.load(args.model_path)
    df_out = df.copy()
    df_out["prediction"] = m.predict(X)
    df_out.to_csv(args.output_csv, index=False)
    print(f"✅ Predictions -> {args.output_csv}")

if __name__ == "__main__":
    main()
