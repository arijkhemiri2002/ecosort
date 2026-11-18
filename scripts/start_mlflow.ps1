# Démarre le serveur MLflow local (http://127.0.0.1:5000)
# Usage : .\scripts\start_mlflow.ps1   (laisser ce terminal ouvert)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

if (-not $env:VIRTUAL_ENV) { .\.venv\Scripts\Activate.ps1 }
New-Item -ItemType Directory -Force -Path .mlflow\artifacts | Out-Null

mlflow server `
  --host 127.0.0.1 `
  --port 5000 `
  --backend-store-uri sqlite:///.mlflow/mlflow.db `
  --artifacts-destination ./.mlflow/artifacts