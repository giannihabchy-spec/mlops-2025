import argparse, pandas as pd
from mlops_2025.features.titanic import TitanicFeaturizer
from mlops_2025.models import LogisticModel, RandomForestModel, XGBModel

MODEL_REGISTRY = {
    "logreg": LogisticModel,
    "rf": RandomForestModel,
    "xgb": XGBModel,
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--target", default="Survived")
    ap.add_argument("--model_out", required=True)
    ap.add_argument("--featurizer_out", default="./data/featurizer.joblib")
    ap.add_argument("--model", choices=MODEL_REGISTRY.keys(), default="logreg")
    args = ap.parse_args()

    df = pd.read_csv(args.train_csv)
    Xy = df.drop(columns=[args.target]), df[args.target].astype(int).values

    fe = TitanicFeaturizer().fit(Xy[0])
    X, _ = fe.transform(Xy[0])

    model = MODEL_REGISTRY[args.model]().fit(X, Xy[1])
    model.save(args.model_out)
    fe.save(args.featurizer_out)

    print(f"✅ {args.model} saved -> {args.model_out}")
    print(f"✅ featurizer saved -> {args.featurizer_out}")

if __name__ == "__main__":
    main()
