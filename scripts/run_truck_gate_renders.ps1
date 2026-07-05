$ErrorActionPreference = "Stop"

$Script = "E:\wjz\test1\dataset\dataset_obj\slice_attention_baseline\origindataset\gated_blender_physical.py"
$Model = "E:\wjz\test1\dataset\dataset_obj\slice_attention_baseline\dataset_new\Military_3D_Dataset\05_Military_Truck_SAM\05_Military_Truck_SAM_004_6ba63e06.glb"
$Root = "E:\wjz\test1\dataset\dataset_obj\slice_attention_baseline\dataset_new"

if ($env:BLENDER_LAUNCHER -and (Test-Path -LiteralPath $env:BLENDER_LAUNCHER)) {
    $Blender = $env:BLENDER_LAUNCHER
}
else {
    $Blender = Get-Command "blender-launcher.exe" -ErrorAction SilentlyContinue |
        Select-Object -First 1 -ExpandProperty Source
}
if (-not $Blender) {
    $Blender = Get-ChildItem -Path "E:\" -Recurse -File -Filter "blender-launcher.exe" -ErrorAction SilentlyContinue |
        Select-Object -First 1 -ExpandProperty FullName
}
if (-not $Blender) {
    throw "Cannot find blender-launcher.exe. Set BLENDER_LAUNCHER or add it to PATH."
}
Write-Host "Using Blender: $Blender"

$CommonArgs = @(
    "--single-model", $Model,
    "--target-size", "6.0",
    "--camera-view", "top",
    "--auto-gate-fit", "visible-bounds",
    "--auto-gate-margin", "0.08",
    "--receiver-gate-width", "0.9",
    "--laser-pulse-width", "0.45",
    "--gate-visibility", "emission",
    "--export-depth",
    "--render-device", "cpu"
)

Write-Host "Rendering true 3D target..."
$TrueArgs = @("--background", "--python", $Script, "--") + $CommonArgs + @(
    "--output-root", (Join-Path $Root "truck_true_target_v2"),
    "--target-mode", "physical-3d"
)
& $Blender @TrueArgs
if ($LASTEXITCODE -ne 0) {
    throw "True target render failed with exit code $LASTEXITCODE."
}

Write-Host "Rendering flat false target..."
$FalseArgs = @("--background", "--python", $Script, "--") + $CommonArgs + @(
    "--output-root", (Join-Path $Root "truck_flat_false_silhouette_v2"),
    "--target-mode", "flat-echo",
    "--flat-target-gate-index", "0",
    "--flat-min-response", "0.18"
)
& $Blender @FalseArgs
if ($LASTEXITCODE -ne 0) {
    throw "Flat false target render failed with exit code $LASTEXITCODE."
}

Write-Host "Done."
