# SPDX-License-Identifier: AGPL-3.0-only

[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$buildScript = Join-Path $PSScriptRoot "build-installer.ps1"
$source = Get-Content -LiteralPath $buildScript -Raw

$requiredPatterns = [ordered]@{
    "detect-only branch" = 'if task == "detect":'
    "YOLO26 raw head" = 'head.end2end = False'
    "raw detect export" = 'model.export(format="onnx", imgsz=640, end2end=False)'
    "pose-compatible export" = 'model.export(format="onnx", imgsz=640)'
    "versioned cache recipe" = 'recipe_version=3'
    "cache export mode" = 'export_mode=$exportMode'
    "raw cache identity" = 'raw-detect-end2end-false-v1'
    "fixed PPE label contract" = 'always-all-seven-v2'
    "fixed PPE label order" = '"Gloves", "Person", "Safety_boots", "Vest", "respirador"'
}

foreach ($entry in $requiredPatterns.GetEnumerator()) {
    if (-not $source.Contains($entry.Value, [System.StringComparison]::Ordinal)) {
        throw "Installer ONNX export policy is missing $($entry.Key): $($entry.Value)"
    }
}

[pscustomobject]@{
    assertions = $requiredPatterns.Count
    result = "passed"
}
