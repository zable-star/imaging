param(
  [string]$Python = "E:\ana\envs\pytorch1\python.exe",
  [string]$DatasetRoot = ".\dataset_new\Military_TF_v8_mv4_norm",
  [string]$ExperimentRoot = ".\experiments",
  [int[]]$Seeds = @(42, 332, 2026),
  [int]$Epochs = 20,
  [int]$BatchSize = 8,
  [bool]$UseAmp = $true,
  [switch]$SkipExisting
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$fusionModes = @("mean", "attention_residual")

foreach ($fusion in $fusionModes) {
  $experimentName = "v8_mv4_norm_mixaug_$($fusion)_full_20ep"
  $experimentDir = Join-Path $ExperimentRoot $experimentName
  $args = @(
    ".\run_experiments.py",
    "--experiment-name", $experimentName,
    "--experiment-root", $experimentDir,
    "--dataset-root", $DatasetRoot,
    "--classes", "true3d", "flat_false",
    "--expected-num-slices", "3",
    "--input-mode", "multi",
    "--single-gate-index", "0",
    "--fusion-mode", $fusion,
    "--gaussian-noise-std", "0.02",
    "--poisson-peak", "80",
    "--background-scatter", "0.02",
    "--degradation-probability", "0.5",
    "--seeds"
  )
  foreach ($seed in $Seeds) {
    $args += "$seed"
  }
  $args += @(
    "--epochs", "$Epochs",
    "--batch-size", "$BatchSize",
    "--num-workers", "0",
    "--lr", "0.001",
    "--val-ratio", "0.25",
    "--split-group-by-sample-id",
    "--keep-going",
    "--results-csv", (Join-Path $experimentDir "results.csv"),
    "--aggregate-csv", (Join-Path $experimentDir "aggregate_results.csv")
  )
  if ($UseAmp) {
    $args += "--use-amp"
  }
  if ($SkipExisting) {
    $args += "--skip-existing"
  }

  Write-Host "Training $experimentName"
  & $Python @args
  if ($LASTEXITCODE -ne 0) {
    throw "Training failed: $experimentName"
  }
}

& $Python ".\scripts\evaluate_v8_mv4_fusion_grid.py" `
  --dataset-root $DatasetRoot `
  --experiment-root $ExperimentRoot `
  --batch-size $BatchSize `
  --summary-csv ".\experiments\v8_mv4_norm_mixaug_full_fusion_eval_summary_3seed.csv" `
  --aggregate-csv ".\experiments\v8_mv4_norm_mixaug_full_fusion_eval_aggregate_3seed.csv" `
  --markdown-out ".\writing\v8_mv4_fusion_robustness_report_2026-07-07.md"
if ($LASTEXITCODE -ne 0) {
  throw "Fusion evaluation failed."
}
