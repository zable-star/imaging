param(
    [string]$Python = "E:\ana\envs\pytorch1\python.exe",
    [string]$ProjectRoot = "E:\wjz\test1\dataset\dataset_obj\slice_attention_baseline",
    [int[]]$Seeds = @(42, 332, 2026),
    [int]$MainEpochs = 20,
    [int]$AblationEpochs = 10,
    [int]$ControlEpochs = 10,
    [int]$BatchSize = 8,
    [int]$NumWorkers = 0,
    [bool]$UseAmp = $true,
    [bool]$SkipExisting = $true,
    [bool]$KeepGoing = $true,
    [string[]]$Stages = @("main", "ablation"),
    [string]$PretrainedModelPath = "experiments\six_class_attention_residual_seedmatched\six_class_attention_residual_seed42\slice_attention_model.pth",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Resolve-ProjectPath {
    param([string]$PathValue)
    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return $PathValue
    }
    return Join-Path $ProjectRoot $PathValue
}

function Test-DatasetOrWarn {
    param([string]$DatasetRoot)
    $FullPath = Resolve-ProjectPath $DatasetRoot
    if (-not (Test-Path -LiteralPath $FullPath)) {
        Write-Warning "Skip missing dataset: $FullPath"
        return $false
    }
    return $true
}

function Invoke-LocalExperiment {
    param(
        [string]$ExperimentName,
        [string]$DatasetRoot,
        [string[]]$Classes,
        [int]$Epochs,
        [string]$FusionMode = "attention_residual",
        [string]$InputMode = "multi",
        [int]$SingleGateIndex = 0,
        [string[]]$ExtraArgs = @()
    )

    if (-not (Test-DatasetOrWarn $DatasetRoot)) {
        return
    }

    $ExperimentRoot = "experiments\$ExperimentName"
    $Runner = Join-Path $ProjectRoot "run_military_transfer_experiments.py"
    $RunnerArgs = @(
        "--classes"
    ) + $Classes + @(
        "--",
        "--experiment-name", $ExperimentName,
        "--experiment-root", $ExperimentRoot,
        "--dataset-root", $DatasetRoot,
        "--fusion-mode", $FusionMode,
        "--input-mode", $InputMode,
        "--single-gate-index", ([string]$SingleGateIndex),
        "--split-group-by-sample-id",
        "--seeds"
    ) + $Seeds + @(
        "--epochs", ([string]$Epochs),
        "--batch-size", ([string]$BatchSize),
        "--num-workers", ([string]$NumWorkers),
        "--results-csv", "$ExperimentRoot\results.csv",
        "--aggregate-csv", "$ExperimentRoot\aggregate_results.csv"
    )

    if ($UseAmp) {
        $RunnerArgs += "--use-amp"
    }
    if ($SkipExisting) {
        $RunnerArgs += "--skip-existing"
    }
    if ($KeepGoing) {
        $RunnerArgs += "--keep-going"
    }
    if ($DryRun) {
        $RunnerArgs += "--dry-run"
    }
    $RunnerArgs += $ExtraArgs

    Write-Host ""
    Write-Host "=== $ExperimentName ==="
    Write-Host "$Python $Runner $($RunnerArgs -join ' ')"
    & $Python $Runner @RunnerArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Experiment failed: $ExperimentName"
    }
}

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python does not exist: $Python"
}
if (-not (Test-Path -LiteralPath $ProjectRoot)) {
    throw "ProjectRoot does not exist: $ProjectRoot"
}

Set-Location $ProjectRoot
Write-Host "ProjectRoot: $ProjectRoot"
Write-Host "Python: $Python"
Write-Host "Stages: $($Stages -join ', ')"
Write-Host "Seeds: $($Seeds -join ', ')"
Write-Host "BatchSize: $BatchSize  NumWorkers: $NumWorkers  UseAmp: $UseAmp"

$MilitaryClasses = @("01_Main_Battle_Tank", "02_Fighter_Jet", "03_Attack_Helicopter")
$TrueFalseClasses = @("true3d", "flat_false")
$MilitaryGated = "dataset_new\Military_3D_Gated_Selected44"
$HardProjection = "dataset_new\Military_TrueFalse_Selected44_hard_projection"
$HardProjectionHistMatched = "dataset_new\Military_TrueFalse_Selected44_hard_projection_hist_classgate_matched"
$HardProjectionHistAreaClipmax = "dataset_new\Military_TrueFalse_Selected44_hard_projection_hist_area_clipmax_classgate_matched"
$RectMatched = "dataset_new\Military_TrueFalse_Selected44_hard_rect_overlap_mean_classgate_matched"
$ForegroundMatched = "dataset_new\Military_TrueFalse_Selected44_hard_rect_overlap_foreground_classgate_matched"
$P99Matched = "dataset_new\Military_TrueFalse_Selected44_hard_rect_overlap_p99_classgate_matched"

if ($Stages -contains "smoke") {
    Invoke-LocalExperiment `
        -ExperimentName "localgpu_smoke_truefalse_rect_matched_2ep" `
        -DatasetRoot $RectMatched `
        -Classes $TrueFalseClasses `
        -Epochs 2 `
        -ExtraArgs @("--lr", "0.0003")
}

if ($Stages -contains "main") {
    Invoke-LocalExperiment `
        -ExperimentName "localgpu_truefalse_rect_matched_full_${MainEpochs}ep" `
        -DatasetRoot $RectMatched `
        -Classes $TrueFalseClasses `
        -Epochs $MainEpochs

    Invoke-LocalExperiment `
        -ExperimentName "localgpu_truefalse_hard_projection_full_${MainEpochs}ep" `
        -DatasetRoot $HardProjection `
        -Classes $TrueFalseClasses `
        -Epochs $MainEpochs

    Invoke-LocalExperiment `
        -ExperimentName "localgpu_military3class_scratch_${MainEpochs}ep" `
        -DatasetRoot $MilitaryGated `
        -Classes $MilitaryClasses `
        -Epochs $MainEpochs
}

if ($Stages -contains "transfer") {
    $ResolvedPretrained = Resolve-ProjectPath $PretrainedModelPath
    if (Test-Path -LiteralPath $ResolvedPretrained) {
        Invoke-LocalExperiment `
            -ExperimentName "localgpu_military3class_transfer_frozen_${MainEpochs}ep" `
            -DatasetRoot $MilitaryGated `
            -Classes $MilitaryClasses `
            -Epochs $MainEpochs `
            -ExtraArgs @("--pretrained-model-path", $ResolvedPretrained, "--freeze-encoder", "--lr", "0.0003")

        Invoke-LocalExperiment `
            -ExperimentName "localgpu_military3class_transfer_finetune_${MainEpochs}ep" `
            -DatasetRoot $MilitaryGated `
            -Classes $MilitaryClasses `
            -Epochs $MainEpochs `
            -ExtraArgs @("--pretrained-model-path", $ResolvedPretrained, "--lr", "0.0001")
    }
    else {
        Write-Warning "Skip transfer experiments because pretrained model was not found: $ResolvedPretrained"
    }
}

if ($Stages -contains "ablation") {
    foreach ($Gate in 0, 1, 2) {
        Invoke-LocalExperiment `
            -ExperimentName "localgpu_truefalse_rect_matched_single_gate${Gate}_${AblationEpochs}ep" `
            -DatasetRoot $RectMatched `
            -Classes $TrueFalseClasses `
            -Epochs $AblationEpochs `
            -InputMode "single-gate" `
            -SingleGateIndex $Gate

        Invoke-LocalExperiment `
            -ExperimentName "localgpu_truefalse_hard_projection_single_gate${Gate}_${AblationEpochs}ep" `
            -DatasetRoot $HardProjection `
            -Classes $TrueFalseClasses `
            -Epochs $AblationEpochs `
            -InputMode "single-gate" `
            -SingleGateIndex $Gate
    }
}

if ($Stages -contains "controls") {
    Invoke-LocalExperiment `
        -ExperimentName "localgpu_truefalse_rect_foreground_matched_full_${ControlEpochs}ep" `
        -DatasetRoot $ForegroundMatched `
        -Classes $TrueFalseClasses `
        -Epochs $ControlEpochs

    Invoke-LocalExperiment `
        -ExperimentName "localgpu_truefalse_rect_p99_matched_full_${ControlEpochs}ep" `
        -DatasetRoot $P99Matched `
        -Classes $TrueFalseClasses `
        -Epochs $ControlEpochs

    Invoke-LocalExperiment `
        -ExperimentName "localgpu_truefalse_rect_foreground_matched_gate1_${ControlEpochs}ep" `
        -DatasetRoot $ForegroundMatched `
        -Classes $TrueFalseClasses `
        -Epochs $ControlEpochs `
        -InputMode "single-gate" `
        -SingleGateIndex 1

    Invoke-LocalExperiment `
        -ExperimentName "localgpu_truefalse_rect_p99_matched_gate1_${ControlEpochs}ep" `
        -DatasetRoot $P99Matched `
        -Classes $TrueFalseClasses `
        -Epochs $ControlEpochs `
        -InputMode "single-gate" `
        -SingleGateIndex 1
}

if ($Stages -contains "hist") {
    Invoke-LocalExperiment `
        -ExperimentName "localgpu_truefalse_hard_projection_hist_full_${MainEpochs}ep" `
        -DatasetRoot $HardProjectionHistMatched `
        -Classes $TrueFalseClasses `
        -Epochs $MainEpochs

    foreach ($Gate in 0, 1, 2) {
        Invoke-LocalExperiment `
            -ExperimentName "localgpu_truefalse_hard_projection_hist_single_gate${Gate}_${MainEpochs}ep" `
            -DatasetRoot $HardProjectionHistMatched `
            -Classes $TrueFalseClasses `
            -Epochs $MainEpochs `
            -InputMode "single-gate" `
            -SingleGateIndex $Gate
    }
}

if ($Stages -contains "geom") {
    Invoke-LocalExperiment `
        -ExperimentName "localgpu_truefalse_hard_projection_hist_area_clipmax_full_${MainEpochs}ep" `
        -DatasetRoot $HardProjectionHistAreaClipmax `
        -Classes $TrueFalseClasses `
        -Epochs $MainEpochs

    foreach ($Gate in 0, 1, 2) {
        Invoke-LocalExperiment `
            -ExperimentName "localgpu_truefalse_hard_projection_hist_area_clipmax_single_gate${Gate}_${MainEpochs}ep" `
            -DatasetRoot $HardProjectionHistAreaClipmax `
            -Classes $TrueFalseClasses `
            -Epochs $MainEpochs `
            -InputMode "single-gate" `
            -SingleGateIndex $Gate
    }
}

if (-not $DryRun) {
    & $Python (Join-Path $ProjectRoot "scripts\collect_paper_experiment_report.py") `
        --experiment-root "experiments" `
        --name-prefix "localgpu_" `
        --output-csv "experiments\localgpu_combined_results.csv" `
        --output-md "writing\localgpu_training_report_2026-07-06.md"
}

Write-Host ""
Write-Host "Local GPU experiment script finished."
