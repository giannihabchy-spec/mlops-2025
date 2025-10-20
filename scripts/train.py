import argparse
import pandas as pd
import numpy as np
from sklearn import ensemble, gaussian_process, linear_model, naive_bayes, neighbors, svm, tree, discriminant_analysis
from sklearn.model_selection import train_test_split, cross_val_score
from joblib import dump

from featurize import create_pipeline

def get_algorithms():
    return [
        # Ensemble
        ensemble.AdaBoostClassifier(),
        ensemble.BaggingClassifier(),
        ensemble.ExtraTreesClassifier(),
        ensemble.GradientBoostingClassifier(),
        ensemble.RandomForestClassifier(),
        # Gaussian Processes
        gaussian_process.GaussianProcessClassifier(),
        # GLM-ish
        linear_model.LogisticRegressionCV(max_iter=1000),
        linear_model.PassiveAggressiveClassifier(),
        linear_model.RidgeClassifierCV(),
        linear_model.SGDClassifier(loss="log_loss", max_iter=2000),
        linear_model.Perceptron(),
        # Naive Bayes
        naive_bayes.BernoulliNB(),
        naive_bayes.GaussianNB(),
        # KNN
        neighbors.KNeighborsClassifier(),
        # SVM
        svm.SVC(probability=True),
        svm.NuSVC(probability=True),
        svm.LinearSVC(),
        # Trees
        tree.DecisionTreeClassifier(),
        tree.ExtraTreeClassifier(),
        # Discriminant Analysis
        discriminant_analysis.LinearDiscriminantAnalysis(),
        discriminant_analysis.QuadraticDiscriminantAnalysis(),
        # XGBoost
    ]

def main():
    ap = argparse.ArgumentParser(description="Train and compare a bunch of classifiers with your featurizer.")
    ap.add_argument("-i","--input", required=True, help="Input CSV (raw data)")
    ap.add_argument("-y","--target", required=True, help="Target column")
    ap.add_argument("--cv", type=int, default=5, help="CV folds (default: 5)")
    ap.add_argument("--test_size", type=float, default=0.2, help="Holdout size (default: 0.2)")
    ap.add_argument("--save_best", default=None, help="Path to save best model (joblib)")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    if args.target not in df.columns:
        raise ValueError(f"Target '{args.target}' not in columns.")

    y = df[args.target].copy()
    X = df.drop(columns=[args.target])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, stratify=y if len(y.unique())>1 else None, random_state=42)

    algos = get_algorithms()
    names, means = [], []

    for algo in algos:
        pipe = create_pipeline(algo)
        scores = cross_val_score(pipe, X_train, y_train, cv=args.cv, n_jobs=-1)
        names.append(algo.__class__.__name__)
        means.append(scores.mean())

    order = np.argsort(means)[::-1]
    print("\nCV leaderboard (mean accuracy):")
    for i in order:
        print(f"{names[i]:30s} {means[i]:.4f}")

    best_algo = algos[order[0]]
    best_pipe = create_pipeline(best_algo).fit(X_train, y_train)
    test_acc = best_pipe.score(X_test, y_test)
    print(f"\nBest: {best_algo.__class__.__name__} | Holdout accuracy: {test_acc:.4f}")

    if args.save_best:
        dump(best_pipe, args.save_best)
        print(f"Saved best model to: {args.save_best}")

if __name__ == "__main__":
    main()
