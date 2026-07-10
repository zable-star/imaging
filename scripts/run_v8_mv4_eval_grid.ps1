param(
  [string]$Python = "E:\ana\envs\pytorch1\python.exe",
  [string]$DatasetRoot = ".\dataset_new\Military_TF_v8_mv4_norm",
  [string]$ExperimentRoot = ".\experiments",
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

$conditions = @(
  @{
    Name = "clean"
    Gaussian = 0.0
    Poisson = 0.0
    Background = 0.0
  },
  @{
    Name = "light_noise_g002_p80_b002"
    Gaussian = 0.02
    Poisson = 80.0
    Background = 0.02
  },
  @{
    Name = "strong_noise_g005_p30_b005"
    Gaussian = 0.05
    Poisson = 30.0
    Background = 0.05
  }
)

foreach ($config in $configs) {
  foreach ($seed in $Seeds) {
    $runName = "$($config.TrainRoot)_seed$seed"
    $modelPath = Join-Path $ExperimentRoot (Join-Path $config.TrainRoot (Join-Path $runName "slice_attention_model.pth"))
    if (-not (Test-Path -LiteralPath $modelPath)) {
      throw "Missing checkpoint: $modelPath"
    }

    foreach ($condition in $conditions) {
      $artifactName = "eval_v8_mv4_$($config.EvalName)_seed$seed`_$($condition.Name)"
      $artifactDir = Join-Path $ExperimentRoot $artifactName

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
        "--gaussian-noise-std", "$($condition.Gaussian)",
        "--poisson-peak", "$($condition.Poisson)",
        "--background-scatter", "$($condition.Background)",
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
}

& $Python ".\scripts\summarize_eval_grid.py" `
  --eval-root $ExperimentRoot `
  --name-prefix "eval_v8_mv4_" `
  --summary-csv ".\experiments\v8_mv4_norm_mixaug_attention_eval_summary_3seed.csv" `
  --aggregate-csv ".\experiments\v8_mv4_norm_mixaug_attention_eval_aggregate_3seed.csv" `
  --markdown-out ".\writing\v8_mv4_multiview_robustness_report_2026-07-07.md"
if ($LASTEXITCODE -ne 0) {
  throw "Summary failed."
}
