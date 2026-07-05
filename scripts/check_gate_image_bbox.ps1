$ErrorActionPreference = "Stop"

param(
    [string]$ImagePath = "E:\wjz\test1\dataset\dataset_obj\slice_attention_baseline\dataset_new\truck_flat_false_silhouette_v2\05_Military_Truck_SAM\05_Military_Truck_SAM_05_Military_Truck_SAM_004_6ba63e06_gate_0.png",
    [int]$Threshold = 5
)

Add-Type -AssemblyName System.Drawing

$bitmap = [System.Drawing.Bitmap]::new($ImagePath)
$minX = $bitmap.Width
$minY = $bitmap.Height
$maxX = -1
$maxY = -1
$count = 0

for ($y = 0; $y -lt $bitmap.Height; $y++) {
    for ($x = 0; $x -lt $bitmap.Width; $x++) {
        $pixel = $bitmap.GetPixel($x, $y)
        if ($pixel.R -gt $Threshold -or $pixel.G -gt $Threshold -or $pixel.B -gt $Threshold) {
            if ($x -lt $minX) { $minX = $x }
            if ($x -gt $maxX) { $maxX = $x }
            if ($y -lt $minY) { $minY = $y }
            if ($y -gt $maxY) { $maxY = $y }
            $count++
        }
    }
}

$bitmap.Dispose()

if ($count -eq 0) {
    Write-Host "No foreground pixels above threshold $Threshold."
    exit 0
}

$touchesBorder = ($minX -eq 0 -or $minY -eq 0 -or $maxX -eq 223 -or $maxY -eq 223)
Write-Host "Foreground bbox: x=$minX..$maxX, y=$minY..$maxY, pixels=$count"
Write-Host "Touches border: $touchesBorder"
