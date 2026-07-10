param(
    [string]$ProjectRoot = "E:\wjz\test1\dataset\dataset_obj\slice_attention_baseline",
    [string]$InputRoot = "E:\wjz\test1\dataset\dataset_obj\slice_attention_baseline\dataset_new\Military_3D_Selected44",
    [string]$OutputRoot = "E:\wjz\test1\dataset\dataset_obj\slice_attention_baseline\dataset_new\Military_3D_Gated_Selected44",
    [string]$RenderDevice = "cpu",
    [string]$CameraView = "top",
    [string]$TargetMode = "physical-3d",
    [double]$TargetSize = 6.0,
    [double]$ReceiverGateWidth = 0.9,
    [double]$LaserPulseWidth = 0.45,
    [double]$AutoGateMargin = 0.08,
    [int]$FlatTargetGateIndex = 0,
    [double]$FlatMinResponse = 0.18,
    [double]$FlatEchoGain = 1.0,
    [ValidateSet("material-only", "flatten-camera-depth")]
    [string]$FlatGeometryMode = "flatten-camera-depth",
    [ValidateSet("fixed", "round-robin", "hash")]
    [string]$FlatTargetGateIndexMode = "round-robin",
    [double]$TargetReflectance = 1.0,
    [ValidateSet("fixed", "hash-uniform", "hash-log-uniform")]
    [string]$ReflectanceMode = "fixed",
    [double]$ReflectanceMin = 1.0,
    [double]$ReflectanceMax = 1.0,
    [string]$FlatTargetDepth = "",
    [string]$ModelRotationDeg = "0,0,0",
    [int]$ModelsPerClass = 0,
    [switch]$ExportDepth,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$Script = Join-Path $ProjectRoot "origindataset\gated_blender_physical.py"
if (-not (Test-Path -LiteralPath $Script)) {
    throw "Cannot find Blender render script: $Script"
}
if (-not $DryRun -and -not (Test-Path -LiteralPath $InputRoot)) {
    throw "InputRoot does not exist: $InputRoot"
}

if ($env:BLENDER_LAUNCHER -and (Test-Path -LiteralPath $env:BLENDER_LAUNCHER)) {
    $Blender = $env:BLENDER_LAUNCHER
}
elseif ($env:BLENDER_EXE -and (Test-Path -LiteralPath $env:BLENDER_EXE)) {
    $Blender = $env:BLENDER_EXE
}
else {
    $Blender = Get-Command "blender-launcher.exe" -ErrorAction SilentlyContinue |
        Select-Object -First 1 -ExpandProperty Source
}
if (-not $Blender) {
    $Blender = Get-Command "blender.exe" -ErrorAction SilentlyContinue |
        Select-Object -First 1 -ExpandProperty Source
}
if (-not $Blender) {
    foreach ($Dir in Get-ChildItem -Path "E:\" -Directory -ErrorAction SilentlyContinue) {
        foreach ($ExeName in @("blender.exe", "blender-launcher.exe")) {
            $Candidate = Join-Path $Dir.FullName $ExeName
            if (Test-Path -LiteralPath $Candidate) {
                $Blender = $Candidate
                break
            }
        }
        if ($Blender) {
            break
        }
    }
}
if (-not $Blender -and $DryRun) {
    $Blender = "<BLENDER_EXE_OR_LAUNCHER>"
}
if (-not $Blender) {
    throw "Cannot find Blender. Set BLENDER_LAUNCHER or BLENDER_EXE."
}

$ArgsList = @(
    "--background",
    "--python", $Script,
    "--",
    "--input-root", $InputRoot,
    "--output-root", $OutputRoot,
    "--target-size", ([string]$TargetSize),
    "--camera-view", $CameraView,
    "--auto-gate-fit", "visible-bounds",
    "--auto-gate-margin", ([string]$AutoGateMargin),
    "--receiver-gate-width", ([string]$ReceiverGateWidth),
    "--laser-pulse-width", ([string]$LaserPulseWidth),
    "--gate-visibility", "emission",
    "--render-device", $RenderDevice,
    "--target-mode", $TargetMode,
    "--models-per-class", ([string]$ModelsPerClass),
    "--target-reflectance", ([string]$TargetReflectance),
    "--reflectance-mode", $ReflectanceMode,
    "--reflectance-min", ([string]$ReflectanceMin),
    "--reflectance-max", ([string]$ReflectanceMax),
    "--model-rotation-deg", $ModelRotationDeg
)

if ($TargetMode -eq "flat-echo") {
    $ArgsList += @(
        "--flat-target-gate-index", ([string]$FlatTargetGateIndex),
        "--flat-min-response", ([string]$FlatMinResponse),
        "--flat-echo-gain", ([string]$FlatEchoGain),
        "--flat-geometry-mode", $FlatGeometryMode,
        "--flat-target-gate-index-mode", $FlatTargetGateIndexMode
    )
    if ($FlatTargetDepth -ne "") {
        $ArgsList += @("--flat-target-depth", $FlatTargetDepth)
    }
}

if ($ExportDepth) {
    $ArgsList += "--export-depth"
}

Write-Host "Using Blender: $Blender"
Write-Host "InputRoot: $InputRoot"
Write-Host "OutputRoot: $OutputRoot"
Write-Host "Command:"
Write-Host "$Blender $($ArgsList -join ' ')"

if ($DryRun) {
    Write-Host "Dry run only; not rendering."
    exit 0
}

& $Blender @ArgsList
if ($LASTEXITCODE -ne 0) {
    throw "Selected military gate rendering failed with exit code $LASTEXITCODE."
}

Write-Host "Done."
