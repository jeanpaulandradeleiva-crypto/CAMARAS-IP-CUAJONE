# SPDX-License-Identifier: AGPL-3.0-only

Set-StrictMode -Version Latest

$script:CuajoneParityReceiptSchema = Join-Path $PSScriptRoot "..\..\contracts\v1\parity-receipt.schema.json"
$script:CuajoneParityMaximumAge = [TimeSpan]::FromDays(7)
$script:CuajoneParityFutureSkew = [TimeSpan]::FromMinutes(5)
$script:CuajoneParityStages = @(
    "contracts-defaults",
    "preprocess-letterbox",
    "detections-keypoints-canonicalization",
    "tracking-ppe-fall-determinism",
    "canonical-events",
    "authorized-engine-video-end-to-end"
)

function Assert-ProductionParityReceipt(
    [string]$Path,
    [string]$ExpectedSourceCommit,
    [string]$ExpectedContractVersion = "1.0.0"
) {
    if ($ExpectedContractVersion -cne "1.0.0" -or
        $ExpectedSourceCommit -notmatch '^[0-9a-f]{40}$') {
        throw "Release gate supports contract 1.0.0 and a full lowercase source commit only"
    }
    if ([string]::IsNullOrWhiteSpace($Path) -or
        -not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Production Release requires a parity receipt: $Path"
    }
    $fullPath = [System.IO.Path]::GetFullPath($Path)
    if (-not $fullPath.StartsWith("D:\", [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Parity receipt must remain on D: $fullPath"
    }
    if (-not (Test-Path -LiteralPath $script:CuajoneParityReceiptSchema -PathType Leaf)) {
        throw "Parity receipt schema was not found: $script:CuajoneParityReceiptSchema"
    }
    try {
        $schemaValid = Test-Json -LiteralPath $fullPath `
            -SchemaFile $script:CuajoneParityReceiptSchema -ErrorAction Stop
    } catch {
        throw "Parity receipt does not satisfy the shared schema: $($_.Exception.Message)"
    }
    if (-not $schemaValid) {
        throw "Parity receipt does not satisfy the shared schema"
    }

    # Preserve the RFC3339 string instead of allowing ConvertFrom-Json to coerce it to DateTime.
    $receipt = Get-Content -LiteralPath $fullPath -Raw | ConvertFrom-Json -DateKind String
    if ($receipt.receipt_version -ne 1 -or
        $receipt.contract_version -cne $ExpectedContractVersion -or
        $receipt.source_commit -cne $ExpectedSourceCommit -or
        $receipt.scope -cne "authorized-engine-data" -or
        $receipt.passed -ne $true -or
        $receipt.full_model_parity_claimed -ne $true) {
        throw "Parity receipt does not authorize a production Release for this contract and source commit"
    }
    if ($receipt.generated_at -notmatch '^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,7})?Z$') {
        throw "Parity receipt generated_at must be an RFC3339 UTC timestamp ending in Z"
    }
    $parsedTime = [DateTimeOffset]::MinValue
    $timestampStyles = [Globalization.DateTimeStyles]::AssumeUniversal -bor `
        [Globalization.DateTimeStyles]::AdjustToUniversal
    if (-not [DateTimeOffset]::TryParse(
            $receipt.generated_at,
            [Globalization.CultureInfo]::InvariantCulture,
            $timestampStyles,
            [ref]$parsedTime)) {
        throw "Parity receipt generated_at is not a valid timestamp"
    }
    $now = [DateTimeOffset]::UtcNow
    if ($parsedTime -gt $now.Add($script:CuajoneParityFutureSkew)) {
        throw "Parity receipt generated_at is beyond the allowed five-minute future skew"
    }
    if ($parsedTime -lt $now.Subtract($script:CuajoneParityMaximumAge)) {
        throw "Parity receipt is older than the seven-day Release validity window"
    }

    if ($receipt.authorization_reference -notmatch '^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$') {
        throw "Parity receipt requires an approved authorization reference"
    }
    $approvedInputs = @($receipt.approved_inputs)
    if ($approvedInputs.Count -lt 2) {
        throw "Parity receipt requires at least two approved input identities and hashes"
    }
    $inputIdentities = [Collections.Generic.HashSet[string]]::new(
        [StringComparer]::Ordinal)
    foreach ($input in $approvedInputs) {
        if ($input.identity -notmatch '^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$' -or
            $input.sha256 -cnotmatch '^[0-9a-f]{64}$' -or
            -not $inputIdentities.Add([string]$input.identity)) {
            throw "Approved parity inputs require unique identities and lowercase SHA-256 hashes"
        }
    }

    if ($receipt.tracker_profiles.production_sim -cne "native-iou" -or
        $receipt.tracker_profiles.experimental_live -cne "ultralytics-bytetrack-not-equivalent") {
        throw "Parity receipt must preserve the explicit native-IoU versus ByteTrack distinction"
    }
    $expectedTolerances = [ordered]@{
        box_absolute = 0.0001
        keypoint_absolute = 0.0001
        confidence_absolute = 0.00001
    }
    foreach ($field in $expectedTolerances.Keys) {
        $value = [double]$receipt.numeric_tolerances.PSObject.Properties[$field].Value
        if (-not [double]::IsFinite($value) -or $value -ne $expectedTolerances[$field]) {
            throw "Parity receipt numeric tolerance changed: $field"
        }
    }

    $stages = @($receipt.stages)
    if ($stages.Count -ne $script:CuajoneParityStages.Count -or
        (($stages.name -join '|') -cne ($script:CuajoneParityStages -join '|'))) {
        throw "Parity receipt must contain the exact six ordered production stages"
    }
    foreach ($stage in $stages) {
        if ($stage.status -cne "passed" -or [long]$stage.comparisons -le 0) {
            throw "Production parity stage did not pass with comparisons: $($stage.name)"
        }
        $evidence = @($stage.evidence)
        if ($evidence.Count -eq 0) {
            throw "Production parity stage has no hashed evidence: $($stage.name)"
        }
        foreach ($item in $evidence) {
            if ($item.identity -notmatch '^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$' -or
                $item.sha256 -cnotmatch '^[0-9a-f]{64}$') {
                throw "Production parity stage has invalid evidence: $($stage.name)"
            }
        }
        $failuresProperty = $stage.PSObject.Properties["failures"]
        if ($null -ne $failuresProperty -and [long]$failuresProperty.Value -ne 0) {
            throw "Production parity stage reports failures: $($stage.name)"
        }
    }
    $endToEnd = $stages[-1]
    if ($null -eq $endToEnd.PSObject.Properties["failures"] -or
        [long]$endToEnd.failures -ne 0 -or
        $endToEnd.authorization_reference -cne $receipt.authorization_reference) {
        throw "Authorized end-to-end stage must report zero failures under the receipt authorization"
    }

    $expectedEventFields = @(
        "id", "source", "type", "time", "subject", "frame_id",
        "monotonic_timestamp_ms", "track_id", "status", "confidence", "evidence"
    )
    $expectedEventOrder = @("frame_id", "track_id", "type", "id")
    if (($receipt.event_normalization.fields -join "|") -cne ($expectedEventFields -join "|") -or
        ($receipt.event_normalization.order_by -join "|") -cne ($expectedEventOrder -join "|") -or
        $receipt.event_normalization.confidence_digits -ne 6) {
        throw "Parity receipt event normalization contract changed"
    }
    $receipt
}
