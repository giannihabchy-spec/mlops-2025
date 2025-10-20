import argparse
import pandas as pd
from joblib import load
import sys

def main():
    ap = argparse.ArgumentParser(description="Run inference with a saved sklearn Pipeline.")
    ap.add_argument("-m","--model", required=True, help="Path to .joblib saved pipeline")
    ap.add_argument("-i","--input", required=True, help="CSV with raw features (no target)")
    ap.add_argument("-o","--output", required=True, help="Where to write predictions CSV")
    ap.add_argument("--id_col", default=None, help="Optional ID column to carry through")
    ap.add_argument("--proba", action="store_true", help="Also output class probabilities (if supported)")
    args = ap.parse_args()

    pipe = load(args.model)
    df = pd.read_csv(args.input)

    out = pd.DataFrame()
    if args.id_col:
        if args.id_col not in df.columns:
            sys.exit(f"ID column '{args.id_col}' not found in input.")
        out[args.id_col] = df[args.id_col]

    y_pred = pipe.predict(df)
    out["prediction"] = y_pred

    if args.proba:
        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(df)
            classes = getattr(pipe, "classes_", None)
            if classes is None:
                try:
                    classes = pipe[-1].classes_
                except Exception:
                    classes = [f"class_{i}" for i in range(proba.shape[1])]
            for i, cls in enumerate(classes):
                out[f"proba_{cls}"] = proba[:, i]
        else:
            sys.exit("This model doesn't support predict_proba. Refit with a probabilistic classifier or drop --proba.")

    out.to_csv(args.output, index=False)
    print(f"Wrote: {args.output}")

if __name__ == "__main__":
    main()
