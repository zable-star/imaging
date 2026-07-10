param(
    [string]$Python = "E:\ana\envs\pytorch1\python.exe",
    [string]$DatasetRoot = "dataset_new\Military_TrueFalse_Selected44_blender_flat_rr_gain2_min035_v4",
    [string]$ExperimentRoot = "experiments",
    [int[]]$Seeds = @(42, 332, 2026),
    [int]$Epochs = 20,
    [int]$BatchSize = 8,
    [int]$NumWorkers = 0,
    [double]$Lr = 0.001,
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
        $Name = "localgpu_blender_flat_rr_gain2_min035_v4_full_${Epochs}ep_seed${Seed}"
    }
    else {
        $Name = "localgpu_blender_flat_rr_gain2_min035_v4_gate${Gate}_${Epochs}ep_seed${Seed}"
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
        "--fusion-mode", "attention",
        "--epochs", ([string]$Epochs),
        "--batch-size", ([string]$BatchSize),
        "--num-workers", ([string]$NumWorkers),
        "--lr", ([string]$Lr),
        "--val-ratio", "0.25",
        "--split-group-by-sample-id",
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
    Invoke-Train -InputMode "multi" -Gate 0 -Seed $Seed
    foreach ($Gate in 0, 1, 2) {
        Invoke-Train -InputMode "single-gate" -Gate $Gate -Seed $Seed
    }
}

Write-Host "Done."
