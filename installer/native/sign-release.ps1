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
    [switch]$AllowInternalPilotTrust = ($env:CUAJONE_ALLOW_INTERNAL_PILOT_TRUST -ceq "1"),
    [string]$PilotRootCertificatePath = $env:CUAJONE_PILOT_ROOT_CER,
    [switch]$VerifyOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Assert-File([string]$Path, [string]$Description) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Description was not found: $Path"
    }
}

function Assert-InternalPilotSignature(
    [string]$Path,
    [System.Management.Automation.Signature]$Signature,
    [string]$ExpectedThumbprint,
    [string]$RootCertificatePath
) {
    Assert-File $RootCertificatePath "Internal pilot root public certificate"
    if ([System.IO.Path]::GetExtension($RootCertificatePath) -cne ".cer") {
        throw "Internal pilot root must be supplied as a public .cer file"
    }
    if ($ExpectedThumbprint -notmatch '^[0-9A-Fa-f]{40}$') {
        throw "Internal pilot verification requires CUAJONE_CERTIFICATE_SHA1"
    }
    if ($null -eq $Signature.SignerCertificate -or
        $Signature.SignerCertificate.Thumbprint -cne $ExpectedThumbprint) {
        throw "Authenticode signer does not match the configured internal pilot leaf certificate"
    }
    if ($null -eq $Signature.TimeStamperCertificate) {
        throw "Internal pilot signature is missing its RFC 3161 timestamp"
    }

    if ($Signature.Status -ne [System.Management.Automation.SignatureStatus]::Valid) {
        if ($Signature.Status -notin @(
            [System.Management.Automation.SignatureStatus]::NotTrusted,
            [System.Management.Automation.SignatureStatus]::UnknownError
        )) {
            throw "Internal pilot signature has a non-trust failure: $($Signature.Status)"
        }

        $defaultChain = [System.Security.Cryptography.X509Certificates.X509Chain]::new()
        try {
            $defaultChain.ChainPolicy.RevocationMode = [System.Security.Cryptography.X509Certificates.X509RevocationMode]::NoCheck
            $defaultChain.Build($Signature.SignerCertificate) | Out-Null
            $defaultStatuses = @(
                $defaultChain.ChainStatus |
                    ForEach-Object { $_.Status } |
                    Sort-Object -Unique
            )
            if ($defaultStatuses.Count -ne 1 -or
                $defaultStatuses[0] -ne [System.Security.Cryptography.X509Certificates.X509ChainStatusFlags]::UntrustedRoot) {
                $statusNames = @($defaultStatuses | ForEach-Object { $_.ToString() }) -join ", "
                throw "Internal pilot signer has chain failures beyond its untrusted private root: $statusNames"
            }
        }
        finally {
            $defaultChain.Dispose()
        }
    }

    $root = [System.Security.Cryptography.X509Certificates.X509Certificate2]::new(
        [System.IO.File]::ReadAllBytes((Resolve-Path -LiteralPath $RootCertificatePath).Path)
    )
    try {
        if ($root.HasPrivateKey) {
            throw "Internal pilot root input must not contain a private key"
        }
        $constraints = $root.Extensions |
            Where-Object { $_ -is [System.Security.Cryptography.X509Certificates.X509BasicConstraintsExtension] } |
            Select-Object -First 1
        if ($null -eq $constraints -or -not $constraints.CertificateAuthority) {
            throw "Internal pilot root input is not a CA certificate"
        }
        $ekuOids = @(
            $Signature.SignerCertificate.Extensions |
                Where-Object { $_ -is [System.Security.Cryptography.X509Certificates.X509EnhancedKeyUsageExtension] } |
                ForEach-Object { $_.EnhancedKeyUsages | ForEach-Object { $_.Value } }
        )
        if ($ekuOids -notcontains "1.3.6.1.5.5.7.3.3") {
            throw "Internal pilot signer lacks the Code Signing EKU"
        }

        $chain = [System.Security.Cryptography.X509Certificates.X509Chain]::new()
        try {
            $chain.ChainPolicy.TrustMode = [System.Security.Cryptography.X509Certificates.X509ChainTrustMode]::CustomRootTrust
            $chain.ChainPolicy.CustomTrustStore.Add($root) | Out-Null
            $chain.ChainPolicy.ExtraStore.Add($root) | Out-Null
            $chain.ChainPolicy.RevocationMode = [System.Security.Cryptography.X509Certificates.X509RevocationMode]::NoCheck
            if (-not $chain.Build($Signature.SignerCertificate) -or
                $chain.ChainElements[$chain.ChainElements.Count - 1].Certificate.Thumbprint -cne $root.Thumbprint) {
                $statuses = @($chain.ChainStatus | ForEach-Object { $_.Status.ToString() }) -join ", "
                throw "Internal pilot signer does not chain to the supplied root: $statuses"
            }
        }
        finally {
            $chain.Dispose()
        }
    }
    finally {
        $root.Dispose()
    }

    [pscustomobject]@{
        File = $Path
        SHA256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
        SignatureStatus = "ValidForInternalPilot"
        Signer = $Signature.SignerCertificate.Subject
        SignerThumbprint = $Signature.SignerCertificate.Thumbprint
        TrustScope = "Private pilot root supplied explicitly; not publicly trusted"
        TimestampCertificate = if ($Signature.TimeStamperCertificate) {
            $Signature.TimeStamperCertificate.Subject
        } else {
            $null
        }
    }
}

Assert-File $FilePath "Signable artifact"
Assert-File $SignToolPath "Microsoft signtool"
$target = (Resolve-Path -LiteralPath $FilePath).Path
$extension = [System.IO.Path]::GetExtension($target).ToLowerInvariant()
if ($extension -notin @(".exe", ".msi") -and
    (Split-Path -Leaf $target) -cne "CuajoneHardwareProbeCA.dll") {
    throw "Only the owned executable, hardware-probe custom action, and MSI may be signed: $target"
}

if ($AllowInternalPilotTrust -and [string]::IsNullOrWhiteSpace($PilotRootCertificatePath)) {
    throw "Internal pilot trust requires CUAJONE_PILOT_ROOT_CER or -PilotRootCertificatePath"
}
if ($AllowInternalPilotTrust) {
    if ($CertificateThumbprint -notmatch '^[0-9A-Fa-f]{40}$') {
        throw "Internal pilot trust requires CUAJONE_CERTIFICATE_SHA1"
    }
    if (-not [string]::IsNullOrWhiteSpace($TrustedSigningDlib) -or
        -not [string]::IsNullOrWhiteSpace($TrustedSigningMetadata)) {
        throw "Internal pilot trust supports only the configured Windows certificate-store leaf"
    }
}

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

$verifyOutput = @(& $SignToolPath verify /pa /all /v $target 2>&1)
$verifyExitCode = $LASTEXITCODE
$verifyOutput | ForEach-Object { Write-Host $_ }
$verifyText = $verifyOutput -join [Environment]::NewLine
if ($verifyText -notmatch '(?i)\bsha256\b') {
    throw "Authenticode verification did not report a SHA-256 digest"
}

$signature = Get-AuthenticodeSignature -LiteralPath $target
if ($AllowInternalPilotTrust) {
    if ($verifyExitCode -ne 0 -and
        $signature.Status -eq [System.Management.Automation.SignatureStatus]::Valid) {
        throw "signtool verification failed unexpectedly for a trusted internal pilot signature"
    }
    Assert-InternalPilotSignature $target $signature $CertificateThumbprint $PilotRootCertificatePath
    return
}
if ($verifyExitCode -ne 0 -or
    $signature.Status -ne [System.Management.Automation.SignatureStatus]::Valid) {
    throw "Trusted Authenticode verification failed: signtool=$verifyExitCode status=$($signature.Status)"
}
if ($null -eq $signature.TimeStamperCertificate) {
    throw "Trusted Authenticode signature is missing its RFC 3161 timestamp"
}
if (-not [string]::IsNullOrWhiteSpace($CertificateThumbprint) -and
    $signature.SignerCertificate.Thumbprint -cne $CertificateThumbprint.ToUpperInvariant()) {
    throw "Authenticode signer does not match the configured certificate-store thumbprint"
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
