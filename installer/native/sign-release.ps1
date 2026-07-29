# SPDX-License-Identifier: AGPL-3.0-only

[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$FilePath,

    [string]$SignToolPath = $env:CUAJONE_SIGNTOOL_PATH,
    [string]$TimestampUrl = $env:CUAJONE_TIMESTAMP_URL,
    [string]$CertificateThumbprint = $env:CUAJONE_CERTIFICATE_SHA1,
    [string]$TrustedSigningDlib = $env:CUAJONE_TRUSTED_SIGNING_DLIB,
    [string]$TrustedSigningMetadata = $env:CUAJONE_TRUSTED_SIGNING_METADATA,
    [switch]$VerifyOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Assert-File([string]$Path, [string]$Description) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Description was not found: $Path"
    }
}

Assert-File $FilePath "PE file"
Assert-File $SignToolPath "Microsoft signtool"
$target = (Resolve-Path -LiteralPath $FilePath).Path

if (-not $VerifyOnly) {
    if ([string]::IsNullOrWhiteSpace($TimestampUrl) -or
        $TimestampUrl -notmatch '^https?://') {
        throw "TimestampUrl must be an approved HTTP(S) RFC 3161 endpoint"
    }

    $hasCertificateStore = -not [string]::IsNullOrWhiteSpace($CertificateThumbprint)
    $hasTrustedSigning = -not [string]::IsNullOrWhiteSpace($TrustedSigningDlib) -or
        -not [string]::IsNullOrWhiteSpace($TrustedSigningMetadata)
    if ($hasCertificateStore -eq $hasTrustedSigning) {
        throw "Select exactly one signing mode: certificate-store thumbprint or Trusted Signing dlib plus metadata"
    }

    $arguments = @(
        "sign", "/v", "/fd", "SHA256", "/tr", $TimestampUrl, "/td", "SHA256"
    )
    if ($hasCertificateStore) {
        if ($CertificateThumbprint -notmatch '^[0-9A-Fa-f]{40}$') {
            throw "CertificateThumbprint must be a 40-character SHA-1 certificate-store thumbprint"
        }
        $arguments += @("/sha1", $CertificateThumbprint)
    }
    else {
        Assert-File $TrustedSigningDlib "Trusted Signing dlib"
        Assert-File $TrustedSigningMetadata "Trusted Signing metadata"
        $arguments += @(
            "/dlib", (Resolve-Path -LiteralPath $TrustedSigningDlib).Path,
            "/dmdf", (Resolve-Path -LiteralPath $TrustedSigningMetadata).Path
        )
    }
    $arguments += $target

    & $SignToolPath @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "signtool sign failed with exit code $LASTEXITCODE"
    }
}

& $SignToolPath verify /pa /all /v $target
if ($LASTEXITCODE -ne 0) {
    throw "signtool verify failed with exit code $LASTEXITCODE"
}

$signature = Get-AuthenticodeSignature -LiteralPath $target
if ($signature.Status -ne [System.Management.Automation.SignatureStatus]::Valid) {
    throw "Authenticode signature is not trusted: $($signature.Status)"
}

[pscustomobject]@{
    File = $target
    SHA256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $target).Hash.ToLowerInvariant()
    SignatureStatus = $signature.Status.ToString()
    Signer = $signature.SignerCertificate.Subject
    TimestampCertificate = if ($signature.TimeStamperCertificate) {
        $signature.TimeStamperCertificate.Subject
    } else {
        $null
    }
}
