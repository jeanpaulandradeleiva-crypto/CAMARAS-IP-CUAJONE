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
    throw "Tracking dependencies must remain under the repository-local .tools/native cache"
}

$downloadRoot = Join-Path $ToolRoot "downloads"
$dependencyRoot = Join-Path $ToolRoot "dependencies"
$temporaryRoot = Join-Path $ToolRoot "temp\tracking-dependencies"
$byteTrackCommit = "a865158906f6138465668810a98ffd918d95f9a3"
$eigenCommit = "3147391d946bb4b6c68edd901f2add6ac1f31f8c"
$byteTrackArchive = Join-Path $downloadRoot "byte-track-eigen-$byteTrackCommit.zip"
$eigenArchive = Join-Path $downloadRoot "eigen-$eigenCommit.zip"
$byteTrackTarget = Join-Path $dependencyRoot "byte-track-eigen-$byteTrackCommit"
$eigenTarget = Join-Path $dependencyRoot "eigen-$eigenCommit"
$patchPath = Join-Path $PSScriptRoot "third_party\byte-track-eigen-cuajone.patch"

function Ensure-Directory([string]$Path) {
    $fullPath = [System.IO.Path]::GetFullPath($Path)
    if (-not $fullPath.StartsWith("$fullToolRoot\", [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Generated dependency path escaped .tools/native: $fullPath"
    }
    if (-not (Test-Path -LiteralPath $fullPath -PathType Container)) {
        New-Item -ItemType Directory -Path $fullPath -Force | Out-Null
    }
}

function Get-VerifiedArchive(
    [string]$Uri,
    [string]$Path,
    [string]$ExpectedSha256,
    [string]$Description
) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        Invoke-WebRequest -Uri $Uri -OutFile $Path
    }
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
    if ($actual -cne $ExpectedSha256) {
        throw "$Description archive SHA-256 mismatch: expected $ExpectedSha256, got $actual"
    }
}

function Expand-VerifiedDependency(
    [string]$Archive,
    [string]$Target,
    [string]$ExpectedRootName,
    [string]$Marker,
    [scriptblock]$AfterExpand
) {
    $receiptPath = Join-Path $Target ".cuajone-source-receipt.json"
    $archiveSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $Archive).Hash.ToLowerInvariant()
    $patchSha256 = if (Test-Path -LiteralPath $patchPath -PathType Leaf) {
        (Get-FileHash -Algorithm SHA256 -LiteralPath $patchPath).Hash.ToLowerInvariant()
    } else { $null }
    if ((Test-Path -LiteralPath (Join-Path $Target $Marker) -PathType Leaf) -and
        (Test-Path -LiteralPath $receiptPath -PathType Leaf)) {
        $existingReceipt = Get-Content -LiteralPath $receiptPath -Raw | ConvertFrom-Json
        $patchApplied = $Target -cne $byteTrackTarget -or
            (Get-Content -LiteralPath (Join-Path $Target $Marker) -Raw).Contains("retained_track_count")
        if ($existingReceipt.archive_sha256 -ceq $archiveSha256 -and
            $existingReceipt.patch_sha256 -ceq $patchSha256 -and $patchApplied) {
            return
        }
    }
    if (Test-Path -LiteralPath $Target) {
        Remove-Item -LiteralPath $Target -Recurse -Force
    }
    if (Test-Path -LiteralPath $temporaryRoot) {
        Remove-Item -LiteralPath $temporaryRoot -Recurse -Force
    }
    Ensure-Directory $temporaryRoot
    Expand-Archive -LiteralPath $Archive -DestinationPath $temporaryRoot -Force
    $expanded = Join-Path $temporaryRoot $ExpectedRootName
    if (-not (Test-Path -LiteralPath $expanded -PathType Container)) {
        throw "Verified dependency archive did not contain expected root: $ExpectedRootName"
    }
    & $AfterExpand $expanded
    Move-Item -LiteralPath $expanded -Destination $Target
    $receipt = [ordered]@{
        archive_sha256 = $archiveSha256
        patch_sha256 = $patchSha256
    }
    $receipt | ConvertTo-Json -Compress | Set-Content -LiteralPath $receiptPath -Encoding UTF8 -NoNewline
}

Ensure-Directory $downloadRoot
Ensure-Directory $dependencyRoot
if (-not (Test-Path -LiteralPath $patchPath -PathType Leaf)) {
    throw "ByteTrack project patch was not found: $patchPath"
}

Get-VerifiedArchive `
    "https://codeload.github.com/cj-mills/byte-track-eigen/zip/$byteTrackCommit" `
    $byteTrackArchive `
    "e5a075df5e8b4ed4bb7436ffe7fe0f4cee5c6a6663112d6a1c47a99ffb704d88" `
    "ByteTrack-Eigen 2.1.0"
Get-VerifiedArchive `
    "https://gitlab.com/libeigen/eigen/-/archive/$eigenCommit/eigen-$eigenCommit.zip" `
    $eigenArchive `
    "9eec4ec4e5e459b2f59dbbaa4280e1bb3ee61cccd8a7c0af0321d29d95fece9e" `
    "Eigen 3.4.0"

Expand-VerifiedDependency `
    $byteTrackArchive `
    $byteTrackTarget `
    "byte-track-eigen-$byteTrackCommit" `
    "include\BYTETracker.h" `
    {
        param($expanded)
        $previousCeiling = $env:GIT_CEILING_DIRECTORIES
        $env:GIT_CEILING_DIRECTORIES = $temporaryRoot
        Push-Location $expanded
        try {
            & git apply --no-index --check $patchPath
            if ($LASTEXITCODE -ne 0) { throw "ByteTrack project patch validation failed" }
            & git apply --no-index $patchPath
            if ($LASTEXITCODE -ne 0) { throw "ByteTrack project patch failed" }
        } finally {
            Pop-Location
            $env:GIT_CEILING_DIRECTORIES = $previousCeiling
        }
    }
Expand-VerifiedDependency `
    $eigenArchive `
    $eigenTarget `
    "eigen-$eigenCommit" `
    "Eigen\Dense" `
    { param($expanded) }

Write-Host "Tracking dependencies verified under $dependencyRoot"
