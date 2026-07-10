param(
    [string]$Python = "E:\ana\envs\pytorch1\python.exe",
    [string]$ProjectRoot = "E:\wjz\test1\dataset\dataset_obj\slice_attention_baseline",
    [string]$DatasetRoot = "dataset_new\Military_TrueFalse_Selected44_gain10",
    [int[]]$GateIndices = @(0, 1, 2),
    [int]$Epochs = 10,
    [int]$Seed = 42,
    [int[]]$Seeds = @(),
    [int]$BatchSize = 8,
    [string]$ExperimentTag = "gain10",
    [int]$MaxGateIndex = 2
)

$ErrorActionPreference = "Stop"
$RunSeeds = if ($Seeds.Count -gt 0) { $Seeds } else { @($Seed) }

foreach ($Gate in $GateIndices) {
    if ($Gate -lt 0 -or $Gate -gt $MaxGateIndex) {
        throw "Invalid gate index $Gate. Expected 0..$MaxGateIndex. If passing multiple values, use a PowerShell array such as -GateIndices @(0,1,2)."
    }
    $ExperimentName = "military_truefalse_${ExperimentTag}_single_gate${Gate}_scratch_${Epochs}ep"
    $ExperimentRoot = "experiments\$ExperimentName"
    & $Python (Join-Path $ProjectRoot "run_military_transfer_experiments.py") `
        --classes true3d flat_false `
        -- `
        --experiment-name $ExperimentName `
        --experiment-root $ExperimentRoot `
        --dataset-root $DatasetRoot `
        --fusion-mode attention_residual `
        --input-mode single-gate `
        --single-gate-index $Gate `
        --split-group-by-sample-id `
        --seeds $RunSeeds `
        --epochs $Epochs `
        --batch-size $BatchSize `
        --results-csv "$ExperimentRoot\results.csv" `
        --aggregate-csv "$ExperimentRoot\aggregate_results.csv"
}
