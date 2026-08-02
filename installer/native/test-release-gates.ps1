# SPDX-License-Identifier: AGPL-3.0-only

[CmdletBinding()]
param(
    [string]$TestRoot
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "release-gates.ps1")
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")).Path
$toolRoot = Join-Path $projectRoot ".tools\native"
if ([string]::IsNullOrWhiteSpace($TestRoot)) { $TestRoot = Join-Path $toolRoot "temp\release-gate-test" }
$fullToolRoot = [System.IO.Path]::GetFullPath($toolRoot).TrimEnd('\')
if (-not [System.IO.Path]::GetFullPath($TestRoot).StartsWith("$fullToolRoot\", [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "TestRoot must remain under the repository-local tool root: $TestRoot"
}

function New-TestHash([char]$Character) {
    (([string]$Character * 64) -join "")
}

function New-PositiveReceipt([string]$Commit) {
    $stageNames = @(
        "contracts-defaults", "preprocess-letterbox",
        "detections-keypoints-canonicalization", "tracking-ppe-fall-determinism",
        "canonical-events", "authorized-engine-video-end-to-end"
    )
    $stages = for ($index = 0; $index -lt $stageNames.Count; $index++) {
        $stage = [ordered]@{
            name = $stageNames[$index]
            status = "passed"
            comparisons = 1
            evidence = @([ordered]@{
                kind = if ($index -eq 5) { "authorized-input" } else { "comparison" }
                identity = "stage-$($index + 1)"
                sha256 = New-TestHash ([char]([int][char]'1' + $index))
            })
        }
        if ($index -eq 5) {
            $stage["failures"] = 0
            $stage["authorization_reference"] = "AUTH-QA-001"
        }
        $stage
    }
    [ordered]@{
        receipt_version = 1
        contract_version = "1.0.0"
        source_commit = $Commit
        generated_at = [DateTimeOffset]::UtcNow.AddMinutes(-1).ToString(
            "yyyy-MM-dd'T'HH:mm:ss.ffffff'Z'", [Globalization.CultureInfo]::InvariantCulture)
        scope = "authorized-engine-data"
        authorization_reference = "AUTH-QA-001"
        approved_inputs = @(
            [ordered]@{ identity = "ppe-engine"; sha256 = New-TestHash 'a' },
            [ordered]@{ identity = "authorized-video"; sha256 = New-TestHash 'b' }
        )
        tracker_profiles = [ordered]@{
            production_sim = "byte-track-eigen"
            experimental_live = "ultralytics-bytetrack-not-equivalent"
        }
        numeric_tolerances = [ordered]@{
            box_absolute = 0.0001
            keypoint_absolute = 0.0001
            confidence_absolute = 0.00001
        }
        event_normalization = [ordered]@{
            fields = @("id", "source", "type", "time", "subject", "frame_id", "monotonic_timestamp_ms", "track_id", "status", "confidence", "evidence")
            order_by = @("frame_id", "track_id", "type", "id")
            confidence_digits = 6
        }
        stages = @($stages)
        full_model_parity_claimed = $true
        passed = $true
    }
}

function Copy-Receipt([object]$Receipt) {
    $Receipt | ConvertTo-Json -Depth 10 | ConvertFrom-Json -DateKind String
}

function Write-Receipt([object]$Receipt, [string]$Path) {
    $Receipt | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $Path -Encoding UTF8
}

function Assert-Rejected(
    [string]$Name,
    [object]$PositiveReceipt,
    [string]$Path,
    [string]$Commit,
    [scriptblock]$Mutation
) {
    $candidate = Copy-Receipt $PositiveReceipt
    & $Mutation $candidate
    Write-Receipt $candidate $Path
    try {
        Assert-ProductionParityReceipt $Path $Commit | Out-Null
    } catch {
        return
    }
    throw "Adversarial parity receipt was accepted: $Name"
}

$parent = Split-Path -Parent $TestRoot
if (-not (Test-Path -LiteralPath $parent -PathType Container)) {
    throw "Release gate test parent does not exist: $parent"
}
if (Test-Path -LiteralPath $TestRoot) {
    Remove-Item -LiteralPath $TestRoot -Recurse -Force
}
New-Item -ItemType Directory -Path $TestRoot | Out-Null
try {
    $commit = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    $missingRejected = $false
    try { Assert-ProductionParityReceipt "" $commit | Out-Null } catch { $missingRejected = $true }
    if (-not $missingRejected) { throw "Missing parity receipt passed the production gate" }

    $receipt = New-PositiveReceipt $commit
    $path = Join-Path $TestRoot "receipt.json"
    Write-Receipt $receipt $path
    Assert-ProductionParityReceipt $path $commit | Out-Null

    Assert-Rejected "synthetic assertion" $receipt $path $commit {
        param($value) $value.scope = "synthetic-only"; $value.full_model_parity_claimed = $false
    }
    Assert-Rejected "source commit mismatch" $receipt $path $commit {
        param($value) $value.source_commit = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    }
    Assert-Rejected "assertion-only receipt" $receipt $path $commit {
        param($value)
        $value.PSObject.Properties.Remove("authorization_reference")
        $value.PSObject.Properties.Remove("approved_inputs")
    }
    Assert-Rejected "future timestamp" $receipt $path $commit {
        param($value)
        $value.generated_at = [DateTimeOffset]::UtcNow.AddMinutes(10).ToString("yyyy-MM-dd'T'HH:mm:ss'Z'")
    }
    Assert-Rejected "stale timestamp" $receipt $path $commit {
        param($value)
        $value.generated_at = [DateTimeOffset]::UtcNow.AddDays(-8).ToString("yyyy-MM-dd'T'HH:mm:ss'Z'")
    }
    Assert-Rejected "non-UTC timestamp" $receipt $path $commit {
        param($value) $value.generated_at = [DateTimeOffset]::UtcNow.ToString("yyyy-MM-dd'T'HH:mm:sszzz")
    }
    Assert-Rejected "zero comparisons" $receipt $path $commit {
        param($value) $value.stages[2].comparisons = 0
    }
    Assert-Rejected "missing authorization" $receipt $path $commit {
        param($value) $value.PSObject.Properties.Remove("authorization_reference")
    }
    Assert-Rejected "unknown stage" $receipt $path $commit {
        param($value) $value.stages[5].name = "forged-stage"
    }
    Assert-Rejected "duplicate stage" $receipt $path $commit {
        param($value) $value.stages[5].name = $value.stages[0].name
    }
    Assert-Rejected "missing stage" $receipt $path $commit {
        param($value) $value.stages = @($value.stages[0..4])
    }
    Assert-Rejected "failed stage with receipt assertion" $receipt $path $commit {
        param($value) $value.stages[3].status = "failed"
    }
    Assert-Rejected "receipt pass inconsistent with stages" $receipt $path $commit {
        param($value) $value.passed = $false
    }
    Assert-Rejected "invalid evidence hash" $receipt $path $commit {
        param($value) $value.stages[1].evidence[0].sha256 = "not-a-hash"
    }
    Assert-Rejected "changed numeric tolerance" $receipt $path $commit {
        param($value) $value.numeric_tolerances.box_absolute = 0.5
    }
    Assert-Rejected "duplicate approved input identity" $receipt $path $commit {
        param($value) $value.approved_inputs[1].identity = $value.approved_inputs[0].identity
    }

    function Assert-FastGateRejected([string]$Name, [scriptblock]$Test) {
        $rejected = $false
        try {
            & $Test
        } catch {
            $rejected = $true
        }
        if (-not $rejected) {
            throw "Fast iteration adversarial gate was accepted: $Name"
        }
    }

    if ((Get-CabinetCompressionLevel $false) -cne "high") { throw "Release must keep high cabinet compression" }
    if ((Get-CabinetCompressionLevel $true) -cne "low") { throw "FastPreview must use low cabinet compression" }
    $fullScope = Get-AcceptanceScope
    $fastScope = Get-AcceptanceScope -FastPreview
    $stageOnlyScope = Get-AcceptanceScope -StageOnly
    if ($fullScope -cne "MSI database, administrative extraction, launcher/runtime PE subsystem, loader, --help, and hardware probe only; no install, engines, cameras, preflight, or inference") {
        throw "Full preview acceptance scope changed"
    }
    if ($fastScope -ceq $fullScope -or $stageOnlyScope -ceq $fullScope -or $stageOnlyScope -ceq $fastScope) {
        throw "FastPreview and StageOnly acceptance scopes must be distinct from the full scope"
    }
    if ($fastScope -notmatch 'administrative extraction') {
        throw "FastPreview scope must honestly disclose the skipped administrative extraction"
    }
    if ($stageOnlyScope -notmatch 'no MSI') {
        throw "StageOnly scope must disclose that no MSI was produced"
    }
    $fastMetadata = Get-FastIterationMetadata -FastPreview
    $releaseMetadata = Get-FastIterationMetadata
    $stageOnlyMetadata = Get-FastIterationMetadata -StageOnly
    if ($fastMetadata.fastPreview -ne $true -or $fastMetadata.stageOnly -ne $false -or $fastMetadata.cabinetCompressionLevel -cne "low") {
        throw "FastPreview receipt metadata is wrong"
    }
    if ($releaseMetadata.fastPreview -ne $false -or $releaseMetadata.stageOnly -ne $false -or $releaseMetadata.cabinetCompressionLevel -cne "high") {
        throw "Release receipt metadata must stay unmarked and high-compression"
    }
    if ($stageOnlyMetadata.stageOnly -ne $true -or $stageOnlyMetadata.fastPreview -ne $false) {
        throw "StageOnly receipt metadata is wrong"
    }
    Assert-FastIterationFlags -BuildMode Preview
    Assert-FastIterationFlags -FastPreview -BuildMode Preview
    Assert-FastIterationFlags -StageOnly -BuildMode Preview
    Assert-FastIterationFlags -FastPreview -StageOnly -BuildMode Preview
    Assert-FastGateRejected "FastPreview with Release" { Assert-FastIterationFlags -FastPreview -BuildMode Release }
    Assert-FastGateRejected "StageOnly with Release" { Assert-FastIterationFlags -StageOnly -BuildMode Release }
    Assert-FastGateRejected "FastPreview and StageOnly with Release" { Assert-FastIterationFlags -FastPreview -StageOnly -BuildMode Release }

    [pscustomobject]@{
        positiveCases = 1
        adversarialCases = 17
        fastIterationPositiveCases = 13
        fastIterationAdversarialCases = 3
        result = "passed"
    }
} finally {
    Remove-Item -LiteralPath $TestRoot -Recurse -Force
}
