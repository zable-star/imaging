param(
    [string]$ProjectRoot = "E:\wjz\test1\dataset\dataset_obj\slice_attention_baseline",
    [string]$InputRoot = "E:\wjz\test1\dataset\dataset_obj\slice_attention_baseline\dataset_new\Military_3D_Selected44",
    [string]$TrueOutputRoot = "E:\wjz\test1\dataset\dataset_obj\slice_attention_baseline\dataset_new\Military_3D_Gated_Selected44_blender_refl_overlap_w15_m012_v8_mv4",
    [string]$FalseOutputRoot = "E:\wjz\test1\dataset\dataset_obj\slice_attention_baseline\dataset_new\Military_FlatFalse_Selected44_blender_refl_overlap_w15_m012_v8_mv4",
    [string]$CombinedOutputRoot = "E:\wjz\test1\dataset\dataset_obj\slice_attention_baseline\dataset_new\Military_TrueFalse_Selected44_blender_refl_overlap_w15_m012_v8_mv4",
    [string]$NormalizedOutputRoot = "E:\wjz\test1\dataset\dataset_obj\slice_attention_baseline\dataset_new\Military_TrueFalse_Selected44_blender_refl_overlap_w15_m012_v8_mv4_per_gate_maxnorm",
    [double[]]$ViewRotationsZ = @(0, 90, 180, 270),
    [string]$RenderDevice = "cpu",
    [int]$ModelsPerClass = 0,
    [double]$TargetSize = 6.0,
    [double]$ReceiverGateWidth = 1.5,
    [double]$LaserPulseWidth = 0.45,
    [double]$AutoGateMargin = 0.12,
    [double]$FlatMinResponse = 0.0,
    [double]$FlatEchoGain = 2.0,
    [string]$ReflectanceMode = "hash-log-uniform",
    [double]$ReflectanceMin = 0.6,
    [double]$ReflectanceMax = 2.8,
    [switch]$SkipRender,
    [switch]$SkipBuild,
    [switch]$SkipNormalize,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$RenderScript = Join-Path $ProjectRoot "scripts\render_selected_military_gates.ps1"
$BuildScript = Join-Path $ProjectRoot "dataset_new\build_multiview_true_false_dataset.py"
$NormalizeScript = Join-Path $ProjectRoot "dataset_new\normalize_gate_dataset.py"
$Python = "E:\ana\envs\pytorch1\python.exe"

if (-not (Test-Path -LiteralPath $RenderScript)) {
    throw "Cannot find render script: $RenderScript"
}
if (-not (Test-Path -LiteralPath $BuildScript)) {
    throw "Cannot find multi-view build script: $BuildScript"
}
if (-not (Test-Path -LiteralPath $NormalizeScript)) {
    throw "Cannot find normalization script: $NormalizeScript"
}
if (-not (Test-Path -LiteralPath $Python)) {
    $Python = "python"
}

function Format-ViewLabel([double]$RotationZ) {
    $rounded = [int][math]::Round($RotationZ)
    return ("view_z{0:D3}" -f $rounded)
}

if (-not $SkipRender) {
    foreach ($RotationZ in $ViewRotationsZ) {
        $ViewLabel = Format-ViewLabel $RotationZ
        $Rotation = "0,0,$RotationZ"
        $TrueViewRoot = Join-Path $TrueOutputRoot $ViewLabel
        $FalseViewRoot = Join-Path $FalseOutputRoot $ViewLabel

        Write-Host "Rendering true 3D view $ViewLabel..."
        $TrueArgs = @(
            "-ExecutionPolicy", "Bypass",
            "-File", $RenderScript,
            "-ProjectRoot", $ProjectRoot,
            "-InputRoot", $InputRoot,
            "-OutputRoot", $TrueViewRoot,
            "-RenderDevice", $RenderDevice,
            "-CameraView", "top",
            "-TargetMode", "physical-3d",
            "-TargetSize", ([string]$TargetSize),
            "-ReceiverGateWidth", ([string]$ReceiverGateWidth),
            "-LaserPulseWidth", ([string]$LaserPulseWidth),
            "-AutoGateMargin", ([string]$AutoGateMargin),
            "-ReflectanceMode", $ReflectanceMode,
            "-ReflectanceMin", ([string]$ReflectanceMin),
            "-ReflectanceMax", ([string]$ReflectanceMax),
            "-ModelRotationDeg", $Rotation,
            "-ModelsPerClass", ([string]$ModelsPerClass)
        )
        if ($DryRun) {
            $TrueArgs += "-DryRun"
        }
        & powershell @TrueArgs
        if ($LASTEXITCODE -ne 0) {
            throw "True 3D render failed for $ViewLabel with exit code $LASTEXITCODE."
        }

        Write-Host "Rendering flat false view $ViewLabel..."
        $FalseArgs = @(
            "-ExecutionPolicy", "Bypass",
            "-File", $RenderScript,
            "-ProjectRoot", $ProjectRoot,
            "-InputRoot", $InputRoot,
            "-OutputRoot", $FalseViewRoot,
            "-RenderDevice", $RenderDevice,
            "-CameraView", "top",
            "-TargetMode", "flat-echo",
            "-TargetSize", ([string]$TargetSize),
            "-ReceiverGateWidth", ([string]$ReceiverGateWidth),
            "-LaserPulseWidth", ([string]$LaserPulseWidth),
            "-AutoGateMargin", ([string]$AutoGateMargin),
            "-FlatMinResponse", ([string]$FlatMinResponse),
            "-FlatEchoGain", ([string]$FlatEchoGain),
            "-FlatGeometryMode", "flatten-camera-depth",
            "-FlatTargetGateIndexMode", "round-robin",
            "-ReflectanceMode", $ReflectanceMode,
            "-ReflectanceMin", ([string]$ReflectanceMin),
            "-ReflectanceMax", ([string]$ReflectanceMax),
            "-ModelRotationDeg", $Rotation,
            "-ModelsPerClass", ([string]$ModelsPerClass)
        )
        if ($DryRun) {
            $FalseArgs += "-DryRun"
        }
        & powershell @FalseArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Flat false render failed for $ViewLabel with exit code $LASTEXITCODE."
        }
    }
}

if ($DryRun) {
    Write-Host "Dry run only; not building or normalizing datasets."
    exit 0
}

if (-not $SkipBuild) {
    & $Python $BuildScript `
        --true-root $TrueOutputRoot `
        --false-root $FalseOutputRoot `
        --output-root $CombinedOutputRoot `
        --expected-num-slices 3 `
        --expected-views $ViewRotationsZ.Count `
        --overwrite
    if ($LASTEXITCODE -ne 0) {
        throw "Multi-view true/false dataset build failed with exit code $LASTEXITCODE."
    }
}

if (-not $SkipNormalize) {
    & $Python $NormalizeScript `
        --input-root $CombinedOutputRoot `
        --output-root $NormalizedOutputRoot `
        --mode per-gate-max `
        --target-max 180 `
        --min-source-max 2 `
        --overwrite
    if ($LASTEXITCODE -ne 0) {
        throw "Multi-view normalization failed with exit code $LASTEXITCODE."
    }
}

Write-Host "Done."
Write-Host "TrueOutputRoot: $TrueOutputRoot"
Write-Host "FalseOutputRoot: $FalseOutputRoot"
Write-Host "CombinedOutputRoot: $CombinedOutputRoot"
Write-Host "NormalizedOutputRoot: $NormalizedOutputRoot"
