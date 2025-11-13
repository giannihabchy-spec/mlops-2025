import argparse
from mlops_2025.preprocessing.titanic import TitanicPreprocessor

def main():
    parser = argparse.ArgumentParser(description="Preprocess Titanic dataset")
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--test_path", required=True)
    parser.add_argument("--output_train", required=True)
    parser.add_argument("--output_test", required=True)
    args = parser.parse_args()

    pre = TitanicPreprocessor()
    train, test = pre.preprocess(args.train_path, args.test_path)
    pre.save(train, test, args.output_train, args.output_test)

    print("✅ Preprocessing complete.")
    print(f"Train saved to: {args.output_train}")
    print(f"Test saved to: {args.output_test}")

if __name__ == "__main__":
    main()
