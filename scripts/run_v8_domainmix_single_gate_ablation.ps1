[CmdletBinding(PositionalBinding = $false)]
param(
  [string]$Python = "E:\ana\envs\pytorch1\python.exe",
  [string]$DatasetRoot = ".\dataset_new\Military_TF_v8_mv4_norm_plus_hardv3_mild",
  [string]$ExperimentRoot = ".\experiments",
  [string[]]$Seeds = @("42", "332", "2026"),
  [int]$Epochs = 20,
  [int]$BatchSize = 8,
  [bool]$UseAmp = $true,
  [switch]$DryRun
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

$configs = @(
  @{ Name = "gate0"; Gate = 0 },
  @{ Name = "gate1"; Gate = 1 },
  @{ Name = "gate2"; Gate = 2 }
)

foreach ($config in $configs) {
  $experimentName = "v8_mv4_domainmix_norm_hardv3_strongaug_attention_$($config.Name)_$($Epochs)ep"
  $experimentRootForConfig = Join-Path $ExperimentRoot $experimentName
  $args = @(
    ".\run_experiments.py",
    "--experiment-name", $experimentName,
    "--experiment-root", $experimentRootForConfig,
    "--dataset-root", $DatasetRoot,
    "--classes", "true3d", "flat_false",
    "--expected-num-slices", "3",
    "--input-mode", "single-gate",
    "--single-gate-index", "$($config.Gate)",
    "--fusion-mode", "attention",
    "--gaussian-noise-std", "0.05",
    "--poisson-peak", "30",
    "--background-scatter", "0.05",
    "--degradation-probability", "0.5",
    "--seeds"
  )
  foreach ($seed in $SeedValues) {
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
    "--results-csv", (Join-Path $experimentRootForConfig "results.csv"),
    "--aggregate-csv", (Join-Path $experimentRootForConfig "aggregate_results.csv")
  )
  if ($UseAmp) {
    $args += "--use-amp"
  }

  $commandForDisplay = "$Python " + ($args -join " ")
  if ($DryRun) {
    Write-Host $commandForDisplay
    continue
  }

  Write-Host "Running $experimentName"
  & $Python @args
  if ($LASTEXITCODE -ne 0) {
    throw "Experiment failed: $experimentName"
  }
}
