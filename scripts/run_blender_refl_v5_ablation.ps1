param(
    [string]$Python = "E:\ana\envs\pytorch1\python.exe",
    [string]$DatasetRoot = "dataset_new\Military_TrueFalse_Selected44_blender_refl_rr_gain2_min035_v5",
    [string]$ExperimentRoot = "experiments",
    [string]$ExperimentTag = "blender_refl_rr_gain2_min035_v5",
    [int[]]$Seeds = @(42, 332, 2026),
    [int[]]$Gates = @(0, 1, 2),
    [switch]$FullOnly,
    [switch]$SingleGateOnly,
    [int]$Epochs = 20,
    [int]$BatchSize = 8,
    [int]$NumWorkers = 0,
    [double]$Lr = 0.001,
    [ValidateSet("attention", "mean", "concat", "attention_residual")]
    [string]$FusionMode = "attention",
    [double]$GaussianNoiseStd = 0.0,
    [double]$PoissonPeak = 0.0,
    [double]$BackgroundScatter = 0.0,
    [double]$BackgroundSigma = 24.0,
    [int]$GateAttenuationIndex = -1,
    [double]$GateAttenuationFactor = 1.0,
    [ValidateSet("none", "fixed", "random")]
    [string]$GateDropoutMode = "none",
    [int]$GateDropoutIndex = 0,
    [double]$DegradationProbability = 1.0,
    [switch]$UseAmp,
    [switch]$CudnnBenchmark,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python executable not found: $Python"
}
if (-not (Test-Path -LiteralPath $DatasetRoot)) {
    throw "Dataset root not found: $DatasetRoot"
}

function Invoke-Train {
    param(
        [string]$InputMode,
        [int]$Gate,
        [int]$Seed
    )

    if ($InputMode -eq "multi") {
        $Name = "localgpu_${ExperimentTag}_full_${Epochs}ep_seed${Seed}"
    }
    else {
        $Name = "localgpu_${ExperimentTag}_gate${Gate}_${Epochs}ep_seed${Seed}"
    }

    $Artifact = Join-Path $ExperimentRoot $Name
    $ArgsList = @(
        "train.py",
        "--dataset-root", $DatasetRoot,
        "--artifact-dir", $Artifact,
        "--model-path", (Join-Path $Artifact "best_model.pth"),
        "--classes", "true3d", "flat_false",
        "--expected-num-slices", "3",
        "--input-mode", $InputMode,
        "--fusion-mode", $FusionMode,
        "--epochs", ([string]$Epochs),
        "--batch-size", ([string]$BatchSize),
        "--num-workers", ([string]$NumWorkers),
        "--lr", ([string]$Lr),
        "--val-ratio", "0.25",
        "--split-group-by-sample-id",
        "--gaussian-noise-std", ([string]$GaussianNoiseStd),
        "--poisson-peak", ([string]$PoissonPeak),
        "--background-scatter", ([string]$BackgroundScatter),
        "--background-sigma", ([string]$BackgroundSigma),
        "--gate-attenuation-index", ([string]$GateAttenuationIndex),
        "--gate-attenuation-factor", ([string]$GateAttenuationFactor),
        "--gate-dropout-mode", $GateDropoutMode,
        "--gate-dropout-index", ([string]$GateDropoutIndex),
        "--degradation-probability", ([string]$DegradationProbability),
        "--seed", ([string]$Seed)
    )

    if ($InputMode -eq "single-gate") {
        $ArgsList += @("--single-gate-index", ([string]$Gate))
    }
    if ($UseAmp) {
        $ArgsList += "--use-amp"
    }
    if ($CudnnBenchmark) {
        $ArgsList += "--cudnn-benchmark"
    }

    Write-Host "Command:"
    Write-Host "$Python $($ArgsList -join ' ')"
    if (-not $DryRun) {
        & $Python @ArgsList
        if ($LASTEXITCODE -ne 0) {
            throw "Training failed: $Name"
        }
    }
}

foreach ($Seed in $Seeds) {
    if (-not $SingleGateOnly) {
        Invoke-Train -InputMode "multi" -Gate 0 -Seed $Seed
    }
    if (-not $FullOnly) {
        foreach ($Gate in $Gates) {
            Invoke-Train -InputMode "single-gate" -Gate $Gate -Seed $Seed
        }
    }
}

Write-Host "Done."
