param(
  [string]$Python = "E:\ana\envs\pytorch1\python.exe",
  [string]$DatasetRoot = ".\dataset_new\Military_TF_v8_mv4_hard_nuisance_v1",
  [string]$ExperimentRoot = ".\experiments",
  [string]$EvalRoot = ".\experiments\v8_mv4_hard_nuisance_v1_eval",
  [string]$ConditionName = "hard_nuisance_v1",
  [int[]]$Seeds = @(42, 332, 2026),
  [int]$BatchSize = 8,
  [bool]$UseAmp = $true
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$configs = @(
  @{
    EvalName = "full"
    TrainRoot = "v8_mv4_norm_mixaug_attention_full_20ep"
    InputMode = "multi"
    SingleGateIndex = 0
  },
  @{
    EvalName = "gate0"
    TrainRoot = "v8_mv4_norm_mixaug_attention_gate0_20ep"
    InputMode = "single-gate"
    SingleGateIndex = 0
  },
  @{
    EvalName = "gate1"
    TrainRoot = "v8_mv4_norm_mixaug_attention_gate1_20ep"
    InputMode = "single-gate"
    SingleGateIndex = 1
  },
  @{
    EvalName = "gate2"
    TrainRoot = "v8_mv4_norm_mixaug_attention_gate2_20ep"
    InputMode = "single-gate"
    SingleGateIndex = 2
  }
)

foreach ($config in $configs) {
  foreach ($seed in $Seeds) {
    $runName = "$($config.TrainRoot)_seed$seed"
    $modelPath = Join-Path $ExperimentRoot (Join-Path $config.TrainRoot (Join-Path $runName "slice_attention_model.pth"))
    if (-not (Test-Path -LiteralPath $modelPath)) {
      throw "Missing checkpoint: $modelPath"
    }

    $artifactName = "eval_v8_mv4_$($config.EvalName)_seed$seed`_$ConditionName"
    $artifactDir = Join-Path $EvalRoot $artifactName
    $args = @(
      ".\scripts\evaluate_gate_model.py",
      "--dataset-root", $DatasetRoot,
      "--model-path", $modelPath,
      "--artifact-dir", $artifactDir,
      "--input-mode", $config.InputMode,
      "--single-gate-index", "$($config.SingleGateIndex)",
      "--fusion-mode", "attention",
      "--batch-size", "$BatchSize",
      "--val-ratio", "0.25",
      "--split-group-by-sample-id",
      "--seed", "$seed",
      "--degradation-probability", "1.0"
    )
    if ($UseAmp) {
      $args += "--use-amp"
    }

    Write-Host "Evaluating $artifactName"
    & $Python @args
    if ($LASTEXITCODE -ne 0) {
      throw "Evaluation failed: $artifactName"
    }
  }
}

& $Python ".\scripts\summarize_eval_grid.py" `
  --eval-root $EvalRoot `
  --name-prefix "eval_v8_mv4_" `
  --summary-csv ".\experiments\v8_mv4_$($ConditionName)_eval_summary_3seed.csv" `
  --aggregate-csv ".\experiments\v8_mv4_$($ConditionName)_eval_aggregate_3seed.csv" `
  --markdown-out ".\writing\v8_mv4_$($ConditionName)_report_2026-07-08.md"
if ($LASTEXITCODE -ne 0) {
  throw "Summary failed."
}
