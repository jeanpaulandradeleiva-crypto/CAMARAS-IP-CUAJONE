# SPDX-License-Identifier: AGPL-3.0-only

[CmdletBinding()]
param(
    [string]$ToolRoot
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$localToolRoot = Join-Path $projectRoot ".tools\native"
if ([string]::IsNullOrWhiteSpace($ToolRoot)) { $ToolRoot = $localToolRoot }
$fullToolRoot = [System.IO.Path]::GetFullPath($ToolRoot).TrimEnd('\')
if ($fullToolRoot -cne [System.IO.Path]::GetFullPath($localToolRoot).TrimEnd('\')) {
    throw "resvg must remain under the repository-local .tools/native cache"
}

$version = "0.47.0"
$archiveUrl = "https://github.com/linebender/resvg/releases/download/v0.47.0/resvg-win64.zip"
$archiveSha256 = "5684e59ceaa53ce720b49efb441b0918ae99d04e8ce3f6f753664524592d67f1"
$downloadRoot = Join-Path $ToolRoot "downloads"
$dependencyRoot = Join-Path $ToolRoot "dependencies"
$temporaryRoot = Join-Path $ToolRoot "temp\resvg-provision"
$archivePath = Join-Path $downloadRoot "resvg-win64.zip"
$target = Join-Path $dependencyRoot "resvg-$version"
$marker = Join-Path $target "resvg.exe"
$receiptPath = Join-Path $target ".cuajone-source-receipt.json"

function Ensure-Directory([string]$Path) {
    $fullPath = [System.IO.Path]::GetFullPath($Path)
    if (-not $fullPath.StartsWith("$fullToolRoot\", [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Generated dependency path escaped .tools/native: $fullPath"
    }
    if (-not (Test-Path -LiteralPath $fullPath -PathType Container)) {
        New-Item -ItemType Directory -Path $fullPath -Force | Out-Null
    }
}

Ensure-Directory $downloadRoot
Ensure-Directory $dependencyRoot

if ((Test-Path -LiteralPath $marker -PathType Leaf) -and
    (Test-Path -LiteralPath $receiptPath -PathType Leaf)) {
    $existingReceipt = Get-Content -LiteralPath $receiptPath -Raw | ConvertFrom-Json
    if ($existingReceipt.archive_sha256 -ceq $archiveSha256 -and
        $existingReceipt.version -ceq $version) {
        Write-Host "resvg $version already provisioned at $target"
        return
    }
}

if (-not (Test-Path -LiteralPath $archivePath -PathType Leaf)) {
    Invoke-WebRequest -Uri $archiveUrl -OutFile $archivePath
}
$actualSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $archivePath).Hash.ToLowerInvariant()
if ($actualSha256 -cne $archiveSha256) {
    throw "resvg $version archive SHA-256 mismatch: expected $archiveSha256, got $actualSha256"
}

if (Test-Path -LiteralPath $target) {
    Remove-Item -LiteralPath $target -Recurse -Force
}
if (Test-Path -LiteralPath $temporaryRoot) {
    Remove-Item -LiteralPath $temporaryRoot -Recurse -Force
}
Ensure-Directory $temporaryRoot
Expand-Archive -LiteralPath $archivePath -DestinationPath $temporaryRoot -Force
$expandedExe = Join-Path $temporaryRoot "resvg.exe"
if (-not (Test-Path -LiteralPath $expandedExe -PathType Leaf)) {
    throw "resvg archive did not contain resvg.exe at its root"
}
Ensure-Directory $target
Move-Item -LiteralPath $expandedExe -Destination $target
$receipt = [ordered]@{
    version = $version
    archive_sha256 = $actualSha256
    source = $archiveUrl
    license = "Apache-2.0 OR MIT"
    note = "Standalone resvg CLI used to rasterize the product SVG icon; no runtime dependencies"
}
$receipt | ConvertTo-Json -Compress | Set-Content -LiteralPath $receiptPath -Encoding UTF8 -NoNewline

Write-Host "resvg $version provisioned under $target"
