param([string]$Model = "logreg")  # logreg | rf | xgb
$ErrorActionPreference = "Stop"

$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
$DATA = Join-Path $ROOT "data"
$ART  = $DATA
$FEAT = Join-Path $ART "featurizer.joblib"
$MODEL_PATH = Join-Path $ART ("model_{0}.joblib" -f $Model)

New-Item -ItemType Directory -Force -Path $DATA | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $DATA "processed") | Out-Null
$env:PYTHONPATH = (Join-Path $ROOT "src")

Write-Host "1) Preprocess"
python (Join-Path $ROOT "scripts\preprocess.py") `
  --train_path (Join-Path $DATA "train.csv") `
  --test_path (Join-Path $DATA "test.csv") `
  --output_train (Join-Path $DATA "processed\processed_train.csv") `
  --output_test (Join-Path $DATA "processed\processed_test.csv")

Write-Host "2) Train ($Model)"
python (Join-Path $ROOT "scripts\train.py") `
  --model $Model `
  --train_csv (Join-Path $DATA "train.csv") `
  --model_out $MODEL_PATH `
  --featurizer_out $FEAT

Write-Host "3) Evaluate"
python (Join-Path $ROOT "scripts\evaluate.py") `
  --eval_csv (Join-Path $DATA "train.csv") `
  --model_path $MODEL_PATH `
  --featurizer_path $FEAT

Write-Host "4) Inference"
python (Join-Path $ROOT "scripts\inference.py") `
  --input_csv (Join-Path $DATA "test.csv") `
  --model_path $MODEL_PATH `
  --featurizer_path $FEAT `
  --output_csv (Join-Path $DATA ("preds_{0}.csv" -f $Model))

Write-Host "✅ Done."
