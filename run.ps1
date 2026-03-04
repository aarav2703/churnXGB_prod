# Runbook: build features -> train -> score
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\run.ps1
# Optional:
#   powershell -ExecutionPolicy Bypass -File .\run.ps1 -SkipBuildFeatures

param(
  [switch]$SkipBuildFeatures
)

$ErrorActionPreference = "Stop"

Write-Host "Repo root:" (Get-Location)

# Ensure PYTHONPATH points to src
$env:PYTHONPATH = "$PWD\src"
Write-Host "PYTHONPATH set to $env:PYTHONPATH"

if (-not $SkipBuildFeatures) {
  Write-Host "`n=== STEP 1: Build Features ==="
  python -m churnxgb.pipeline.build_features
} else {
  Write-Host "`n=== STEP 1: Build Features (SKIPPED) ==="
}

Write-Host "`n=== STEP 2: Train + Log to MLflow + Promote ==="
python -m churnxgb.pipeline.train

Write-Host "`n=== STEP 3: Score + Targets + Drift Monitoring + MLflow Logging ==="
python -m churnxgb.pipeline.score

Write-Host "`nDONE."