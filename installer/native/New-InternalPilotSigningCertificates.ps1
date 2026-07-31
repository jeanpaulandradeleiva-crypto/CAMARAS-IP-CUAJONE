# SPDX-License-Identifier: AGPL-3.0-only

[CmdletBinding()]
param(
    [string]$OutputDirectory,
    [int]$RootValidityYears = 10,
    [int]$LeafValidityYears = 2
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")).Path
$toolRoot = Join-Path $projectRoot ".tools\native"
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) { $OutputDirectory = Join-Path $toolRoot "signing" }

$storeLocation = "Cert:\CurrentUser\My"
$rootSubject = "CN=Cuajone PPE Monitor Internal Pilot Root CA 2026, O=Cuajone PPE Monitor Project"
$leafSubject = "CN=Cuajone PPE Monitor Internal Pilot Code Signing 2026, O=Cuajone PPE Monitor Project"
$rootFriendlyName = "Cuajone PPE Monitor Internal Pilot Root CA 2026"
$leafFriendlyName = "Cuajone PPE Monitor Internal Pilot Code Signing 2026"
$rootCerName = "Cuajone-PPE-Monitor-Internal-Pilot-Root-CA-2026.cer"
$leafCerName = "Cuajone-PPE-Monitor-Internal-Pilot-Code-Signing-2026.cer"
$codeSigningOid = "1.3.6.1.5.5.7.3.3"
$sha256WithRsaOid = "1.2.840.113549.1.1.11"

function Test-DistinguishedName(
    [System.Security.Cryptography.X509Certificates.X509Certificate2]$Certificate,
    [string]$ExpectedSubject
) {
    return $Certificate.Subject -ceq $ExpectedSubject
}

function Get-MatchingCertificate([string]$Subject, [string]$FriendlyName) {
    $matches = @(
        Get-ChildItem -LiteralPath $storeLocation | Where-Object {
            $_.FriendlyName -ceq $FriendlyName -or
                (Test-DistinguishedName $_ $Subject)
        }
    )
    if ($matches.Count -gt 1) {
        throw "Multiple pilot certificates match '$FriendlyName'. Resolve the ambiguity manually; no certificate was replaced."
    }
    if ($matches.Count -eq 1) {
        return $matches[0]
    }
    return $null
}

function Get-BasicConstraints(
    [System.Security.Cryptography.X509Certificates.X509Certificate2]$Certificate
) {
    return $Certificate.Extensions |
        Where-Object { $_ -is [System.Security.Cryptography.X509Certificates.X509BasicConstraintsExtension] } |
        Select-Object -First 1
}

function Get-KeyUsage(
    [System.Security.Cryptography.X509Certificates.X509Certificate2]$Certificate
) {
    return $Certificate.Extensions |
        Where-Object { $_ -is [System.Security.Cryptography.X509Certificates.X509KeyUsageExtension] } |
        Select-Object -First 1
}

function Assert-NonExportableRsaPrivateKey(
    [System.Security.Cryptography.X509Certificates.X509Certificate2]$Certificate,
    [string]$Description
) {
    if (-not $Certificate.HasPrivateKey) {
        throw "$Description does not have a private key in $storeLocation"
    }

    $rsa = [System.Security.Cryptography.X509Certificates.RSACertificateExtensions]::GetRSAPrivateKey(
        $Certificate
    )
    if ($null -eq $rsa) {
        throw "$Description does not use an RSA private key"
    }
    try {
        if ($rsa -is [System.Security.Cryptography.RSACng]) {
            $exportPolicy = $rsa.Key.ExportPolicy
            $exportFlags = [System.Security.Cryptography.CngExportPolicies]::AllowExport -bor
                [System.Security.Cryptography.CngExportPolicies]::AllowPlaintextExport
            if (($exportPolicy -band $exportFlags) -ne 0) {
                throw "$Description has an exportable private key and cannot be reused"
            }
        }
        elseif ($rsa -is [System.Security.Cryptography.RSACryptoServiceProvider]) {
            if ($rsa.CspKeyContainerInfo.Exportable) {
                throw "$Description has an exportable private key and cannot be reused"
            }
        }
        else {
            throw "$Description uses an unsupported RSA key provider"
        }
    }
    finally {
        $rsa.Dispose()
    }
}

function Assert-RootCertificate(
    [System.Security.Cryptography.X509Certificates.X509Certificate2]$Certificate
) {
    if (-not (Test-DistinguishedName $Certificate $rootSubject) -or
        $Certificate.FriendlyName -cne $rootFriendlyName) {
        throw "The existing pilot root identity does not match the required pilot-only identity"
    }
    Assert-NonExportableRsaPrivateKey $Certificate "Pilot root"
    if ($Certificate.NotBefore.ToUniversalTime() -gt [DateTime]::UtcNow -or
        $Certificate.NotAfter.ToUniversalTime() -le [DateTime]::UtcNow.AddDays(30)) {
        throw "The existing pilot root is not currently reusable; it was not replaced"
    }
    if ($Certificate.SignatureAlgorithm.Value -cne $sha256WithRsaOid) {
        throw "The existing pilot root does not use SHA-256 with RSA"
    }
    $constraints = Get-BasicConstraints $Certificate
    if ($null -eq $constraints -or -not $constraints.CertificateAuthority) {
        throw "The existing pilot root is not a CA certificate"
    }
    $usage = Get-KeyUsage $Certificate
    $required = [System.Security.Cryptography.X509Certificates.X509KeyUsageFlags]::KeyCertSign -bor
        [System.Security.Cryptography.X509Certificates.X509KeyUsageFlags]::CrlSign
    if ($null -eq $usage -or ($usage.KeyUsages -band $required) -ne $required) {
        throw "The existing pilot root lacks certificate-signing key usage"
    }
    if ([System.Convert]::ToHexString($Certificate.SubjectName.RawData) -cne
        [System.Convert]::ToHexString($Certificate.IssuerName.RawData)) {
        throw "The existing pilot root is not self-issued"
    }
}

function Assert-LeafCertificate(
    [System.Security.Cryptography.X509Certificates.X509Certificate2]$Certificate,
    [System.Security.Cryptography.X509Certificates.X509Certificate2]$Root
) {
    if (-not (Test-DistinguishedName $Certificate $leafSubject) -or
        $Certificate.FriendlyName -cne $leafFriendlyName) {
        throw "The existing pilot leaf identity does not match the required pilot-only identity"
    }
    Assert-NonExportableRsaPrivateKey $Certificate "Pilot code-signing leaf"
    if ($Certificate.NotBefore.ToUniversalTime() -gt [DateTime]::UtcNow -or
        $Certificate.NotAfter.ToUniversalTime() -le [DateTime]::UtcNow.AddDays(30)) {
        throw "The existing pilot code-signing leaf is not currently reusable; it was not replaced"
    }
    if ($Certificate.SignatureAlgorithm.Value -cne $sha256WithRsaOid) {
        throw "The existing pilot code-signing leaf does not use SHA-256 with RSA"
    }
    if ($Certificate.NotAfter.ToUniversalTime() -gt $Root.NotAfter.ToUniversalTime()) {
        throw "The existing pilot code-signing leaf expires after its pilot root"
    }
    $constraints = Get-BasicConstraints $Certificate
    if ($null -ne $constraints -and $constraints.CertificateAuthority) {
        throw "The existing pilot code-signing leaf is incorrectly marked as a CA"
    }
    $usage = Get-KeyUsage $Certificate
    $digitalSignature = [System.Security.Cryptography.X509Certificates.X509KeyUsageFlags]::DigitalSignature
    if ($null -eq $usage -or ($usage.KeyUsages -band $digitalSignature) -ne $digitalSignature) {
        throw "The existing pilot code-signing leaf lacks DigitalSignature key usage"
    }
    $ekuOids = @(
        $Certificate.Extensions |
            Where-Object { $_ -is [System.Security.Cryptography.X509Certificates.X509EnhancedKeyUsageExtension] } |
            ForEach-Object { $_.EnhancedKeyUsages | ForEach-Object { $_.Value } }
    )
    if ($ekuOids -notcontains $codeSigningOid) {
        throw "The existing pilot code-signing leaf lacks the Code Signing EKU"
    }

    $chain = [System.Security.Cryptography.X509Certificates.X509Chain]::new()
    try {
        $chain.ChainPolicy.TrustMode = [System.Security.Cryptography.X509Certificates.X509ChainTrustMode]::CustomRootTrust
        $chain.ChainPolicy.CustomTrustStore.Add($Root) | Out-Null
        $chain.ChainPolicy.ExtraStore.Add($Root) | Out-Null
        $chain.ChainPolicy.RevocationMode = [System.Security.Cryptography.X509Certificates.X509RevocationMode]::NoCheck
        if (-not $chain.Build($Certificate) -or
            $chain.ChainElements[$chain.ChainElements.Count - 1].Certificate.Thumbprint -cne $Root.Thumbprint) {
            $statuses = @($chain.ChainStatus | ForEach-Object { $_.Status.ToString() }) -join ", "
            throw "The existing pilot leaf does not chain to the pilot root: $statuses"
        }
    }
    finally {
        $chain.Dispose()
    }
}

function Export-PublicCertificate(
    [System.Security.Cryptography.X509Certificates.X509Certificate2]$Certificate,
    [string]$Path
) {
    if (Test-Path -LiteralPath $Path) {
        if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
            throw "Public certificate path is not a file: $Path"
        }
        $existing = [System.Security.Cryptography.X509Certificates.X509Certificate2]::new(
            [System.IO.File]::ReadAllBytes($Path)
        )
        try {
            if ($existing.HasPrivateKey -or $existing.Thumbprint -cne $Certificate.Thumbprint) {
                throw "Existing public certificate does not match and was not replaced: $Path"
            }
        }
        finally {
            $existing.Dispose()
        }
        return "Reused"
    }

    Export-Certificate -Cert $Certificate -FilePath $Path -Type CERT | Out-Null
    return "Exported"
}

if ($RootValidityYears -lt 2) {
    throw "RootValidityYears must be at least 2"
}
if ($LeafValidityYears -lt 1 -or $LeafValidityYears -ge $RootValidityYears) {
    throw "LeafValidityYears must be at least 1 and less than RootValidityYears"
}

$fullOutputDirectory = [System.IO.Path]::GetFullPath($OutputDirectory)
$fullToolRoot = [System.IO.Path]::GetFullPath($toolRoot).TrimEnd('\')
if (-not $fullOutputDirectory.StartsWith("$fullToolRoot\", [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Public certificate output must remain under the repository-local tool root: $fullOutputDirectory"
}
$outputParent = Split-Path -Parent $fullOutputDirectory
if (-not (Test-Path -LiteralPath $outputParent -PathType Container)) {
    throw "Public certificate output parent does not exist: $outputParent"
}
if (-not (Test-Path -LiteralPath $fullOutputDirectory)) {
    New-Item -ItemType Directory -Path $fullOutputDirectory | Out-Null
}
elseif (-not (Test-Path -LiteralPath $fullOutputDirectory -PathType Container)) {
    throw "Public certificate output is not a directory: $fullOutputDirectory"
}

$root = Get-MatchingCertificate $rootSubject $rootFriendlyName
$rootAction = "Reused"
if ($null -eq $root) {
    $notBefore = [DateTime]::Now.AddMinutes(-5)
    $root = New-SelfSignedCertificate `
        -Type Custom `
        -Subject $rootSubject `
        -FriendlyName $rootFriendlyName `
        -CertStoreLocation $storeLocation `
        -Provider "Microsoft Software Key Storage Provider" `
        -KeyAlgorithm RSA `
        -KeyLength 3072 `
        -HashAlgorithm SHA256 `
        -KeyExportPolicy NonExportable `
        -KeyUsage CertSign, CRLSign `
        -NotBefore $notBefore `
        -NotAfter $notBefore.AddYears($RootValidityYears) `
        -TextExtension @("2.5.29.19={critical}{text}ca=1&pathlength=0")
    $rootAction = "Created"
}
Assert-RootCertificate $root

$leaf = Get-MatchingCertificate $leafSubject $leafFriendlyName
$leafAction = "Reused"
if ($null -eq $leaf) {
    $notBefore = [DateTime]::Now.AddMinutes(-5)
    if ($root.NotAfter -le $notBefore.AddYears($LeafValidityYears)) {
        throw "The existing pilot root cannot cover the requested leaf validity; no leaf was created"
    }
    $leaf = New-SelfSignedCertificate `
        -Type Custom `
        -Subject $leafSubject `
        -FriendlyName $leafFriendlyName `
        -Signer $root `
        -CertStoreLocation $storeLocation `
        -Provider "Microsoft Software Key Storage Provider" `
        -KeyAlgorithm RSA `
        -KeyLength 3072 `
        -HashAlgorithm SHA256 `
        -KeyExportPolicy NonExportable `
        -KeyUsage DigitalSignature `
        -NotBefore $notBefore `
        -NotAfter $notBefore.AddYears($LeafValidityYears) `
        -TextExtension @(
            "2.5.29.19={critical}{text}ca=0",
            "2.5.29.37={critical}{text}$codeSigningOid"
        )
    $leafAction = "Created"
}
Assert-LeafCertificate $leaf $root

$rootCerPath = Join-Path $fullOutputDirectory $rootCerName
$leafCerPath = Join-Path $fullOutputDirectory $leafCerName
$rootExportAction = Export-PublicCertificate $root $rootCerPath
$leafExportAction = Export-PublicCertificate $leaf $leafCerPath

[pscustomobject]@{
    Store = $storeLocation
    RootAction = $rootAction
    RootSubject = $root.Subject
    RootThumbprint = $root.Thumbprint
    RootNotAfter = $root.NotAfter.ToUniversalTime().ToString("o")
    RootPublicCertificate = $rootCerPath
    RootPublicCertificateAction = $rootExportAction
    LeafAction = $leafAction
    LeafSubject = $leaf.Subject
    LeafThumbprint = $leaf.Thumbprint
    LeafNotAfter = $leaf.NotAfter.ToUniversalTime().ToString("o")
    LeafPublicCertificate = $leafCerPath
    LeafPublicCertificateAction = $leafExportAction
    CertificateThumbprintEnvironment = "CUAJONE_CERTIFICATE_SHA1=$($leaf.Thumbprint)"
}
