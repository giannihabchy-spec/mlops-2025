import argparse, pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from mlops_2025.features.titanic import TitanicFeaturizer
from mlops_2025.models import LogisticModel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_csv", required=True)
    ap.add_argument("--target", default="Survived")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--featurizer_path", default="./data/featurizer.joblib")
    args = ap.parse_args()

    df = pd.read_csv(args.eval_csv)
    y = df[args.target].astype(int).values
    fe = TitanicFeaturizer.load(args.featurizer_path)
    X, _ = fe.transform(df.drop(columns=[args.target]))

    m = LogisticModel.load(args.model_path)
    yhat = m.predict(X)
    print(f"accuracy={accuracy_score(y,yhat):.4f}  f1={f1_score(y,yhat):.4f}")

if __name__ == "__main__":
    main()
