import argparse
import pandas as pd
from mlops_2025.features import TitanicFeaturizer

def main():
    ap = argparse.ArgumentParser(description="Featurize Titanic-like data into a model-ready matrix.")
    ap.add_argument("--input", "-i", required=True)
    ap.add_argument("--output", "-o", required=True)
    ap.add_argument("--target", "-y", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    y = None
    if args.target and args.target in df.columns:
        y = df[args.target].copy()
        df = df.drop(columns=[args.target])

    X, names = TitanicFeaturizer().transform(df)

    out = pd.DataFrame(X, columns=names)
    if y is not None:
        out[args.target] = y.values
    out.to_csv(args.output, index=False)
    print(f"✅ Wrote {out.shape} to {args.output}")

if __name__ == "__main__":
    main()
