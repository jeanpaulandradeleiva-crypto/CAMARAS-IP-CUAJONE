# SPDX-License-Identifier: AGPL-3.0-only

[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$buildScript = Join-Path $PSScriptRoot "build-installer.ps1"
$source = Get-Content -LiteralPath $buildScript -Raw
$packageSource = Get-Content -LiteralPath (Join-Path $PSScriptRoot "Package.wxs") -Raw

$requiredPatterns = [ordered]@{
    "detect-only branch" = 'if task == "detect":'
    "YOLO26 raw head" = 'head.end2end = False'
    "raw detect export" = 'model.export(format="onnx", imgsz=640, dynamic=True, nms=False, end2end=False)'
    "pose-compatible export" = 'model.export(format="onnx", imgsz=640, dynamic=True, nms=False)'
    "versioned cache recipe" = 'recipe_version=4'
    "cache export mode" = 'export_mode=$exportMode'
    "raw cache identity" = 'raw-detect-dynamic-end2end-false-v2'
    "bounded image sizes" = '"allowed_image_sizes": [640, 768, 960, 1280]'
    "all-size inference validation" = 'for image_size in (640, 768, 960, 1280):'
    "normalized cache receipt hash" = '$exported.onnxSha256 = $manifest.modelSha256'
    "staged PPE ONNX hash" = '$ppeStagedOnnxSha256 = (Get-FileHash -Algorithm SHA256'
    "staged pose ONNX hash" = '$poseStagedOnnxSha256 = (Get-FileHash -Algorithm SHA256'
    "staged ONNX hash verification" = 'Staged ONNX model hash differs from the normalized export cache'
    "metadata uses staged PPE hash" = 'artifactSha256 = $ppeStagedOnnxSha256'
    "metadata uses staged pose hash" = 'artifactSha256 = $poseStagedOnnxSha256'
    "fixed PPE label contract" = 'always-all-seven-v2'
    "fixed PPE label order" = '"Gloves", "Person", "Safety_boots", "Vest", "respirador"'
}

foreach ($entry in $requiredPatterns.GetEnumerator()) {
    if (-not $source.Contains($entry.Value, [System.StringComparison]::Ordinal)) {
        throw "Installer ONNX export policy is missing $($entry.Key): $($entry.Value)"
    }
}

if ($source.Contains("SkipModelBundle", [System.StringComparison]::Ordinal)) {
    throw "Installer still exposes an unsupported model-bundle opt-out"
}
if (-not $packageSource.Contains(
        '<ComponentGroupRef Id="ModelComponents" />',
        [System.StringComparison]::Ordinal) -or
    $packageSource.Contains('Feature Id="ModelsFeature"', [System.StringComparison]::Ordinal)) {
    throw "Models must be mandatory members of MainFeature"
}

[pscustomobject]@{
    assertions = $requiredPatterns.Count
    result = "passed"
}
