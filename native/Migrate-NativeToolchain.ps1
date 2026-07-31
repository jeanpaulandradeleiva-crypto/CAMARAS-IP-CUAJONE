# SPDX-License-Identifier: AGPL-3.0-only

[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$LegacyToolRoot
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$targetRoot = Join-Path $projectRoot ".tools\native"
$sourceRoot = (Resolve-Path -LiteralPath $LegacyToolRoot).Path

if (-not (Test-Path -LiteralPath $sourceRoot -PathType Container)) {
    throw "Legacy native toolchain was not found: $sourceRoot"
}
if (Test-Path -LiteralPath $targetRoot) {
    throw "Repository-local native toolchain already exists: $targetRoot"
}

$targetParent = Split-Path -Parent $targetRoot
if (-not (Test-Path -LiteralPath $targetParent -PathType Container)) {
    New-Item -ItemType Directory -Path $targetParent -Force | Out-Null
}

Move-Item -LiteralPath $sourceRoot -Destination $targetRoot
Write-Host "Native toolchain moved to $targetRoot"
