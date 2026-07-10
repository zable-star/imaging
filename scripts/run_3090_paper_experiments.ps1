param(
    [string]$Python = "E:\ana\envs\pytorch1\python.exe",
    [string]$ProjectRoot = "E:\wjz\test1\dataset\dataset_obj\slice_attention_baseline",
    [int[]]$Seeds = @(42, 332, 2026, 730, 1009),
    [int]$CoreEpochs = 80,
    [int]$AblationEpochs = 40,
    [int]$FusionEpochs = 60,
    [int]$RobustnessEpochs = 40,
    [int]$BatchSize = 32,
    [int]$NumWorkers = 4,
    [bool]$UseAmp = $true,
    [bool]$CudnnBenchmark = $false,
    [bool]$SkipExisting = $true,
    [bool]$KeepGoing = $true,
    [string[]]$Stages = @("core", "ablation", "fusion", "robustness", "controls"),
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

function Invoke-PaperExperiment {
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
    if ($CudnnBenchmark) {
        $RunnerArgs += "--cudnn-benchmark"
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
$RectMatched = "dataset_new\Military_TrueFalse_Selected44_hard_rect_overlap_mean_classgate_matched"
$HardProjection = "dataset_new\Military_TrueFalse_Selected44_hard_projection"
$HardProjectionHistMatched = "dataset_new\Military_TrueFalse_Selected44_hard_projection_hist_classgate_matched"
$HardProjectionHistAreaClipmax = "dataset_new\Military_TrueFalse_Selected44_hard_projection_hist_area_clipmax_classgate_matched"
$MilitaryGated = "dataset_new\Military_3D_Gated_Selected44"
$ForegroundMatched = "dataset_new\Military_TrueFalse_Selected44_hard_rect_overlap_foreground_classgate_matched"
$P99Matched = "dataset_new\Military_TrueFalse_Selected44_hard_rect_overlap_p99_classgate_matched"

if ($Stages -contains "core") {
    Invoke-PaperExperiment `
        -ExperimentName "paper3090_military3class_scratch_${CoreEpochs}ep" `
        -DatasetRoot $MilitaryGated `
        -Classes $MilitaryClasses `
        -Epochs $CoreEpochs `
        -ExtraArgs @("--lr", "0.0003")

    $ResolvedPretrained = Resolve-ProjectPath $PretrainedModelPath
    if (Test-Path -LiteralPath $ResolvedPretrained) {
        Invoke-PaperExperiment `
            -ExperimentName "paper3090_military3class_transfer_frozen_${CoreEpochs}ep" `
            -DatasetRoot $MilitaryGated `
            -Classes $MilitaryClasses `
            -Epochs $CoreEpochs `
            -ExtraArgs @("--pretrained-model-path", $ResolvedPretrained, "--freeze-encoder", "--lr", "0.0003")

        Invoke-PaperExperiment `
            -ExperimentName "paper3090_military3class_transfer_finetune_${CoreEpochs}ep" `
            -DatasetRoot $MilitaryGated `
            -Classes $MilitaryClasses `
            -Epochs $CoreEpochs `
            -ExtraArgs @("--pretrained-model-path", $ResolvedPretrained, "--lr", "0.0001")
    }
    else {
        Write-Warning "Skip transfer experiments because pretrained model was not found: $ResolvedPretrained"
    }

    Invoke-PaperExperiment `
        -ExperimentName "paper3090_truefalse_rect_matched_full_${CoreEpochs}ep" `
        -DatasetRoot $RectMatched `
        -Classes $TrueFalseClasses `
        -Epochs $CoreEpochs

    Invoke-PaperExperiment `
        -ExperimentName "paper3090_truefalse_hard_projection_full_${CoreEpochs}ep" `
        -DatasetRoot $HardProjection `
        -Classes $TrueFalseClasses `
        -Epochs $CoreEpochs
}

if ($Stages -contains "ablation") {
    foreach ($Gate in 0, 1, 2) {
        Invoke-PaperExperiment `
            -ExperimentName "paper3090_truefalse_rect_matched_single_gate${Gate}_${AblationEpochs}ep" `
            -DatasetRoot $RectMatched `
            -Classes $TrueFalseClasses `
            -Epochs $AblationEpochs `
            -InputMode "single-gate" `
            -SingleGateIndex $Gate

        Invoke-PaperExperiment `
            -ExperimentName "paper3090_truefalse_hard_projection_single_gate${Gate}_${AblationEpochs}ep" `
            -DatasetRoot $HardProjection `
            -Classes $TrueFalseClasses `
            -Epochs $AblationEpochs `
            -InputMode "single-gate" `
            -SingleGateIndex $Gate
    }
}

if ($Stages -contains "fusion") {
    foreach ($Fusion in "mean", "attention", "attention_residual", "concat") {
        Invoke-PaperExperiment `
            -ExperimentName "paper3090_truefalse_rect_matched_${Fusion}_${FusionEpochs}ep" `
            -DatasetRoot $RectMatched `
            -Classes $TrueFalseClasses `
            -Epochs $FusionEpochs `
            -FusionMode $Fusion
    }
}

if ($Stages -contains "robustness") {
    Invoke-PaperExperiment `
        -ExperimentName "paper3090_truefalse_rect_matched_noise_bg_poisson_${RobustnessEpochs}ep" `
        -DatasetRoot $RectMatched `
        -Classes $TrueFalseClasses `
        -Epochs $RobustnessEpochs `
        -ExtraArgs @("--gaussian-noise-std", "0.03", "--poisson-peak", "30", "--background-scatter", "0.08")

    Invoke-PaperExperiment `
        -ExperimentName "paper3090_truefalse_rect_matched_random_gate_dropout_${RobustnessEpochs}ep" `
        -DatasetRoot $RectMatched `
        -Classes $TrueFalseClasses `
        -Epochs $RobustnessEpochs `
        -ExtraArgs @("--gate-dropout-mode", "random")

    foreach ($Gate in 0, 1, 2) {
        Invoke-PaperExperiment `
            -ExperimentName "paper3090_truefalse_rect_matched_gate${Gate}_atten035_${RobustnessEpochs}ep" `
            -DatasetRoot $RectMatched `
            -Classes $TrueFalseClasses `
            -Epochs $RobustnessEpochs `
            -ExtraArgs @("--gate-attenuation-index", ([string]$Gate), "--gate-attenuation-factor", "0.35")
    }
}

if ($Stages -contains "controls") {
    Invoke-PaperExperiment `
        -ExperimentName "paper3090_truefalse_rect_foreground_matched_full_${AblationEpochs}ep" `
        -DatasetRoot $ForegroundMatched `
        -Classes $TrueFalseClasses `
        -Epochs $AblationEpochs

    Invoke-PaperExperiment `
        -ExperimentName "paper3090_truefalse_rect_p99_matched_full_${AblationEpochs}ep" `
        -DatasetRoot $P99Matched `
        -Classes $TrueFalseClasses `
        -Epochs $AblationEpochs

    Invoke-PaperExperiment `
        -ExperimentName "paper3090_truefalse_rect_foreground_matched_gate1_${AblationEpochs}ep" `
        -DatasetRoot $ForegroundMatched `
        -Classes $TrueFalseClasses `
        -Epochs $AblationEpochs `
        -InputMode "single-gate" `
        -SingleGateIndex 1

    Invoke-PaperExperiment `
        -ExperimentName "paper3090_truefalse_rect_p99_matched_gate1_${AblationEpochs}ep" `
        -DatasetRoot $P99Matched `
        -Classes $TrueFalseClasses `
        -Epochs $AblationEpochs `
        -InputMode "single-gate" `
        -SingleGateIndex 1
}

if ($Stages -contains "hist") {
    Invoke-PaperExperiment `
        -ExperimentName "paper3090_truefalse_hard_projection_hist_full_${AblationEpochs}ep" `
        -DatasetRoot $HardProjectionHistMatched `
        -Classes $TrueFalseClasses `
        -Epochs $AblationEpochs

    foreach ($Gate in 0, 1, 2) {
        Invoke-PaperExperiment `
            -ExperimentName "paper3090_truefalse_hard_projection_hist_single_gate${Gate}_${AblationEpochs}ep" `
            -DatasetRoot $HardProjectionHistMatched `
            -Classes $TrueFalseClasses `
            -Epochs $AblationEpochs `
            -InputMode "single-gate" `
            -SingleGateIndex $Gate
    }
}

if ($Stages -contains "geom") {
    Invoke-PaperExperiment `
        -ExperimentName "paper3090_truefalse_hard_projection_hist_area_clipmax_full_${AblationEpochs}ep" `
        -DatasetRoot $HardProjectionHistAreaClipmax `
        -Classes $TrueFalseClasses `
        -Epochs $AblationEpochs

    foreach ($Gate in 0, 1, 2) {
        Invoke-PaperExperiment `
            -ExperimentName "paper3090_truefalse_hard_projection_hist_area_clipmax_single_gate${Gate}_${AblationEpochs}ep" `
            -DatasetRoot $HardProjectionHistAreaClipmax `
            -Classes $TrueFalseClasses `
            -Epochs $AblationEpochs `
            -InputMode "single-gate" `
            -SingleGateIndex $Gate
    }
}

if (-not $DryRun) {
    $ReportScript = Join-Path $ProjectRoot "scripts\collect_paper_experiment_report.py"
    if (Test-Path -LiteralPath $ReportScript) {
        & $Python $ReportScript `
            --experiment-root "experiments" `
            --name-prefix "paper3090_" `
            --output-csv "experiments\paper3090_combined_results.csv" `
            --output-md "writing\paper3090_training_report_2026-07-06.md"
    }
}

Write-Host ""
Write-Host "3090 experiment script finished."
