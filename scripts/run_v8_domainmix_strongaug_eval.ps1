[CmdletBinding(PositionalBinding = $false)]
param(
  [string]$Python = "E:\ana\envs\pytorch1\python.exe",
  [string]$ExperimentRoot = ".\experiments",
  [string]$TrainRoot = "v8_mv4_domainmix_norm_hardv3_strongaug_attention_full_20ep",
  [string]$EvalRoot = ".\experiments\v8_mv4_domainmix_norm_hardv3_strongaug_attention_full_eval_3seed",
  [string[]]$Seeds = @("42", "332", "2026"),
  [int]$BatchSize = 8,
  [bool]$UseAmp = $true
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$SeedValues = @()
foreach ($seedText in $Seeds) {
  foreach ($part in "$seedText".Split(",")) {
    $trimmed = $part.Trim()
    if ($trimmed.Length -gt 0) {
      $SeedValues += [int]$trimmed
    }
  }
}
if ($SeedValues.Count -eq 0) {
  throw "No seeds were provided."
}

$conditions = @(
  @{
    Name = "clean"
    Dataset = ".\dataset_new\Military_TF_v8_mv4_norm"
    Gaussian = "0.0"
    Poisson = "0.0"
    Background = "0.0"
  },
  @{
    Name = "light_noise_g002_p80_b002"
    Dataset = ".\dataset_new\Military_TF_v8_mv4_norm"
    Gaussian = "0.02"
    Poisson = "80"
    Background = "0.02"
  },
  @{
    Name = "strong_noise_g005_p30_b005"
    Dataset = ".\dataset_new\Military_TF_v8_mv4_norm"
    Gaussian = "0.05"
    Poisson = "30"
    Background = "0.05"
  },
  @{
    Name = "hard_nuisance_v2"
    Dataset = ".\dataset_new\Military_TF_v8_mv4_hard_nuisance_v2"
    Gaussian = "0.0"
    Poisson = "0.0"
    Background = "0.0"
  },
  @{
    Name = "hard_nuisance_v3_mild"
    Dataset = ".\dataset_new\Military_TF_v8_mv4_hard_nuisance_v3_mild"
    Gaussian = "0.0"
    Poisson = "0.0"
    Background = "0.0"
  }
)

New-Item -ItemType Directory -Force -Path $EvalRoot | Out-Null

foreach ($seed in $SeedValues) {
  $runName = "$TrainRoot`_seed$seed"
  $modelPath = Join-Path $ExperimentRoot (Join-Path $TrainRoot (Join-Path $runName "slice_attention_model.pth"))
  if (-not (Test-Path -LiteralPath $modelPath)) {
    throw "Missing checkpoint: $modelPath"
  }

  foreach ($condition in $conditions) {
    $artifactDir = Join-Path $EvalRoot "eval_v8_mv4_full_seed$seed`_$($condition.Name)"
    $args = @(
      ".\scripts\evaluate_gate_model.py",
      "--dataset-root", $condition.Dataset,
      "--model-path", $modelPath,
      "--artifact-dir", $artifactDir,
      "--classes", "true3d", "flat_false",
      "--expected-num-slices", "3",
      "--input-mode", "multi",
      "--single-gate-index", "0",
      "--fusion-mode", "attention",
      "--batch-size", "$BatchSize",
      "--num-workers", "0",
      "--val-ratio", "0.25",
      "--split-group-by-sample-id",
      "--seed", "$seed",
      "--gaussian-noise-std", $condition.Gaussian,
      "--poisson-peak", $condition.Poisson,
      "--background-scatter", $condition.Background,
      "--degradation-probability", "1.0"
    )
    if ($UseAmp) {
      $args += "--use-amp"
    }

    Write-Host "Evaluating $artifactDir"
    & $Python @args
    if ($LASTEXITCODE -ne 0) {
      throw "Evaluation failed: seed=$seed condition=$($condition.Name)"
    }
  }
}

& $Python ".\scripts\summarize_eval_grid.py" `
  --eval-root $EvalRoot `
  --name-prefix "eval_v8_mv4_" `
  --summary-csv ".\experiments\v8_mv4_domainmix_norm_hardv3_strongaug_attention_full_eval_summary_3seed.csv" `
  --aggregate-csv ".\experiments\v8_mv4_domainmix_norm_hardv3_strongaug_attention_full_eval_aggregate_3seed.csv" `
  --markdown-out ".\writing\v8_mv4_domainmix_strongaug_report_2026-07-08.md"
if ($LASTEXITCODE -ne 0) {
  throw "Summary failed."
}
