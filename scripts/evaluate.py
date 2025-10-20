import argparse
import pandas as pd
from sklearn import ensemble, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import dump

from featurize import create_pipeline

def build_model(which: str):
    which = which.lower()
    if which == "rf":
        return ensemble.RandomForestClassifier(
            criterion="gini",
            n_estimators=1750,
            max_depth=7,
            min_samples_split=6,
            min_samples_leaf=6,
            max_features="sqrt",   
            oob_score=True,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
    if which == "gbdt":
        return ensemble.GradientBoostingClassifier(
            max_depth=1, max_features=None, n_estimators=3,
            random_state=42, warm_start=True
        )
    if which == "ridge":
        return linear_model.RidgeClassifierCV()
    raise ValueError("Unknown model. Use: rf | gbdt | ridge")

def main():
    ap = argparse.ArgumentParser(description="Evaluate a model with a fixed featurization pipeline.")
    ap.add_argument("-i","--input", required=True, help="Input CSV (raw data)")
    ap.add_argument("-y","--target", required=True, help="Target column")
    ap.add_argument("--model", default="rf", choices=["rf","gbdt","ridge"], help="Which classifier to use")
    ap.add_argument("--test_size", type=float, default=0.2, help="Holdout fraction")
    ap.add_argument("--save", default=None, help="Optional path to save the fitted pipeline (.joblib)")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    if args.target not in df.columns:
        raise ValueError(f"Target '{args.target}' not in columns.")

    y = df[args.target].copy()
    X = df.drop(columns=[args.target])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y if y.nunique()>1 else None, random_state=42
    )

    clf = build_model(args.model)
    pipe = create_pipeline(clf)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))

    if args.save:
        dump(pipe, args.save)
        print(f"\nSaved fitted pipeline to: {args.save}")

if __name__ == "__main__":
    main()
