# SPDX-License-Identifier: AGPL-3.0-only

[CmdletBinding()]
param(
    [string]$RootCertificatePath = "D:\DevTools\CuajoneNative\signing\Cuajone-PPE-Monitor-Internal-Pilot-Root-CA-2026.cer",
    [string]$LeafCertificatePath = "D:\DevTools\CuajoneNative\signing\Cuajone-PPE-Monitor-Internal-Pilot-Code-Signing-2026.cer",
    [switch]$ValidateOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$rootSubject = "CN=Cuajone PPE Monitor Internal Pilot Root CA 2026, O=Cuajone PPE Monitor Project"
$leafSubject = "CN=Cuajone PPE Monitor Internal Pilot Code Signing 2026, O=Cuajone PPE Monitor Project"
$codeSigningOid = "1.3.6.1.5.5.7.3.3"
$sha256WithRsaOid = "1.2.840.113549.1.1.11"

function Test-DistinguishedName(
    [System.Security.Cryptography.X509Certificates.X509Certificate2]$Certificate,
    [string]$ExpectedSubject
) {
    return $Certificate.Subject -ceq $ExpectedSubject
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

function Read-PublicCertificate([string]$Path, [string]$Description) {
    if (-not [System.IO.Path]::IsPathFullyQualified($Path)) {
        throw "$Description path must be absolute: $Path"
    }
    if ([System.IO.Path]::GetExtension($Path) -cne ".cer") {
        throw "$Description must be a .cer file: $Path"
    }
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Description was not found: $Path"
    }
    $resolved = (Resolve-Path -LiteralPath $Path).Path
    $certificate = [System.Security.Cryptography.X509Certificates.X509Certificate2]::new(
        [System.IO.File]::ReadAllBytes($resolved)
    )
    if ($certificate.HasPrivateKey) {
        $certificate.Dispose()
        throw "$Description unexpectedly contains a private key"
    }
    return [pscustomobject]@{
        Path = $resolved
        Certificate = $certificate
    }
}

function Import-PublicCertificate([string]$Path, [string]$Store, [string]$Thumbprint) {
    $existing = @(
        Get-ChildItem -LiteralPath $Store | Where-Object { $_.Thumbprint -ceq $Thumbprint }
    )
    if ($existing.Count -gt 1) {
        throw "Multiple copies of certificate $Thumbprint exist in $Store"
    }
    if ($existing.Count -eq 1) {
        return "AlreadyPresent"
    }
    Import-Certificate -FilePath $Path -CertStoreLocation $Store | Out-Null
    $imported = Get-ChildItem -LiteralPath $Store |
        Where-Object { $_.Thumbprint -ceq $Thumbprint } |
        Select-Object -First 1
    if ($null -eq $imported -or $imported.HasPrivateKey) {
        throw "Public certificate import validation failed for $Store"
    }
    return "Imported"
}

$identity = [System.Security.Principal.WindowsIdentity]::GetCurrent()
$principal = [System.Security.Principal.WindowsPrincipal]::new($identity)
if (-not $principal.IsInRole([System.Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw "Administrator privileges are required to modify LocalMachine certificate stores"
}

$rootInput = Read-PublicCertificate $RootCertificatePath "Pilot root public certificate"
$leafInput = Read-PublicCertificate $LeafCertificatePath "Pilot leaf public certificate"
$root = $rootInput.Certificate
$leaf = $leafInput.Certificate
try {
    if (-not (Test-DistinguishedName $root $rootSubject)) {
        throw "Root certificate does not have the required pilot-only subject"
    }
    if (-not (Test-DistinguishedName $leaf $leafSubject)) {
        throw "Leaf certificate does not have the required pilot-only subject"
    }
    if ($root.SignatureAlgorithm.Value -cne $sha256WithRsaOid -or
        $leaf.SignatureAlgorithm.Value -cne $sha256WithRsaOid) {
        throw "Pilot root and leaf certificates must use SHA-256 with RSA"
    }
    $rootConstraints = Get-BasicConstraints $root
    if ($null -eq $rootConstraints -or -not $rootConstraints.CertificateAuthority) {
        throw "Pilot root certificate is not a CA"
    }
    $rootUsage = Get-KeyUsage $root
    $rootRequired = [System.Security.Cryptography.X509Certificates.X509KeyUsageFlags]::KeyCertSign -bor
        [System.Security.Cryptography.X509Certificates.X509KeyUsageFlags]::CrlSign
    if ($null -eq $rootUsage -or ($rootUsage.KeyUsages -band $rootRequired) -ne $rootRequired) {
        throw "Pilot root certificate lacks certificate-signing key usage"
    }
    if ([System.Convert]::ToHexString($root.SubjectName.RawData) -cne
        [System.Convert]::ToHexString($root.IssuerName.RawData)) {
        throw "Pilot root certificate is not self-issued"
    }

    $leafConstraints = Get-BasicConstraints $leaf
    if ($null -ne $leafConstraints -and $leafConstraints.CertificateAuthority) {
        throw "Pilot leaf certificate is incorrectly marked as a CA"
    }
    $leafUsage = Get-KeyUsage $leaf
    $digitalSignature = [System.Security.Cryptography.X509Certificates.X509KeyUsageFlags]::DigitalSignature
    if ($null -eq $leafUsage -or ($leafUsage.KeyUsages -band $digitalSignature) -ne $digitalSignature) {
        throw "Pilot leaf certificate lacks DigitalSignature key usage"
    }
    $ekuOids = @(
        $leaf.Extensions |
            Where-Object { $_ -is [System.Security.Cryptography.X509Certificates.X509EnhancedKeyUsageExtension] } |
            ForEach-Object { $_.EnhancedKeyUsages | ForEach-Object { $_.Value } }
    )
    if ($ekuOids -notcontains $codeSigningOid) {
        throw "Pilot leaf certificate lacks the Code Signing EKU"
    }

    $chain = [System.Security.Cryptography.X509Certificates.X509Chain]::new()
    try {
        $chain.ChainPolicy.TrustMode = [System.Security.Cryptography.X509Certificates.X509ChainTrustMode]::CustomRootTrust
        $chain.ChainPolicy.CustomTrustStore.Add($root) | Out-Null
        $chain.ChainPolicy.ExtraStore.Add($root) | Out-Null
        $chain.ChainPolicy.RevocationMode = [System.Security.Cryptography.X509Certificates.X509RevocationMode]::NoCheck
        if (-not $chain.Build($leaf) -or
            $chain.ChainElements[$chain.ChainElements.Count - 1].Certificate.Thumbprint -cne $root.Thumbprint) {
            $statuses = @($chain.ChainStatus | ForEach-Object { $_.Status.ToString() }) -join ", "
            throw "Pilot leaf does not chain to the supplied pilot root: $statuses"
        }
    }
    finally {
        $chain.Dispose()
    }

    $rootAction = if ($ValidateOnly) {
        "ValidatedOnly"
    } else {
        Import-PublicCertificate $rootInput.Path "Cert:\LocalMachine\Root" $root.Thumbprint
    }
    $leafAction = if ($ValidateOnly) {
        "ValidatedOnly"
    } else {
        Import-PublicCertificate $leafInput.Path "Cert:\LocalMachine\TrustedPublisher" $leaf.Thumbprint
    }

    [pscustomobject]@{
        RootStore = "Cert:\LocalMachine\Root"
        RootAction = $rootAction
        RootSubject = $root.Subject
        RootThumbprint = $root.Thumbprint
        LeafStore = "Cert:\LocalMachine\TrustedPublisher"
        LeafAction = $leafAction
        LeafSubject = $leaf.Subject
        LeafThumbprint = $leaf.Thumbprint
    }
}
finally {
    $root.Dispose()
    $leaf.Dispose()
}
