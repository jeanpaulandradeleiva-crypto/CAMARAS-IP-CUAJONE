# SPDX-License-Identifier: AGPL-3.0-only

[CmdletBinding()]
param(
    [string]$TestRoot
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "payload-policy.ps1")
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")).Path
$toolRoot = Join-Path $projectRoot ".tools\native"
if ([string]::IsNullOrWhiteSpace($TestRoot)) { $TestRoot = Join-Path $toolRoot "temp\payload-policy-test" }
$fullToolRoot = [System.IO.Path]::GetFullPath($toolRoot).TrimEnd('\')
if (-not [System.IO.Path]::GetFullPath($TestRoot).StartsWith("$fullToolRoot\", [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "TestRoot must remain under the repository-local tool root: $TestRoot"
}

$parent = Split-Path -Parent $TestRoot
if (-not (Test-Path -LiteralPath $parent -PathType Container)) {
    throw "Payload policy test parent does not exist: $parent"
}
if (Test-Path -LiteralPath $TestRoot) {
    Remove-Item -LiteralPath $TestRoot -Recurse -Force
}
New-Item -ItemType Directory -Path $TestRoot | Out-Null
try {
    $allowedPaths = @(
        "bin\NexoAIVisionLauncher.exe",
        "bin\NexoAIVision.exe",
        "bin\opencv_world4120.dll",
        "bin\engine-builder\trtexec.exe",
        "bin\engine-builder\ppe.onnx",
        "bin\engine-builder\pose.onnx",
        "bin\engine-builder\nvinfer_plugin_11.dll",
        "docs\README.md",
        "contracts\v1\event.schema.json",
        "licenses\LICENSE.txt",
        "build-metadata.json"
    )
    foreach ($relative in $allowedPaths) {
        $candidate = Join-Path $TestRoot $relative
        $candidateParent = Split-Path -Parent $candidate
        if (-not (Test-Path -LiteralPath $candidateParent)) {
            New-Item -ItemType Directory -Path $candidateParent -Force | Out-Null
        }
        "allowed" | Set-Content -LiteralPath $candidate -Encoding ASCII
    }
    Assert-NoForbiddenPayloadFiles $TestRoot "policy test"
    $forbiddenPaths = @(
        "bin\unexpected_tool.exe",
        "cuajone_native.pyd",
        "python312.dll",
        "python311.dll",
        "python313_d.dll",
        "python3.dll",
        "python.exe",
        "pythonw.exe",
        "python311.zip",
        "module.py",
        "Lib\encodings\aliases.py",
        "site-packages\native-extension.dll",
        "fixtures\frame.json",
        "parity-receipt.json",
        "cvat\client.txt",
        "supervision\adapter.txt",
        "model.engine.manifest.json",
        "model.plan",
        "weights.safetensors",
        "model.onnx",
        "model.pt",
        "model.pth",
        "model.engine",
        "model.bin",
        "data\sample.bin",
        "data\payload.json",
        "dataset.parquet",
        "dataset\sample.arrow",
        "dataset\sample.feather",
        "models\payload.dat"
    )
    foreach ($relative in $forbiddenPaths) {
        $candidate = Join-Path $TestRoot $relative
        $candidateParent = Split-Path -Parent $candidate
        if (-not (Test-Path -LiteralPath $candidateParent)) {
            New-Item -ItemType Directory -Path $candidateParent -Force | Out-Null
        }
        "synthetic" | Set-Content -LiteralPath $candidate -Encoding ASCII
        $failedClosed = $false
        try {
            Assert-NoForbiddenPayloadFiles $TestRoot "policy test"
        } catch {
            $failedClosed = $true
        }
        if (-not $failedClosed) {
            throw "Forbidden payload was accepted: $relative"
        }
        Remove-Item -LiteralPath $candidate -Force
    }
    [pscustomobject]@{
        allowedCases = $allowedPaths.Count
        forbiddenCases = $forbiddenPaths.Count
        result = "passed"
    }
} finally {
    Remove-Item -LiteralPath $TestRoot -Recurse -Force
}
