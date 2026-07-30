# SPDX-License-Identifier: AGPL-3.0-only

[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$MsiPath,

    [Parameter(Mandatory)]
    [string]$RootCertificatePath,

    [Parameter(Mandatory)]
    [string]$LeafCertificatePath,

    [Parameter(Mandatory)]
    [ValidatePattern('^[0-9A-Fa-f]{64}$')]
    [string]$ExpectedMsiSha256,

    [Parameter(Mandatory)]
    [ValidatePattern('^[0-9A-Fa-f]{64}$')]
    [string]$ExpectedRootCertificateSha256,

    [Parameter(Mandatory)]
    [ValidatePattern('^[0-9A-Fa-f]{64}$')]
    [string]$ExpectedLeafCertificateSha256,

    [Parameter(Mandatory)]
    [ValidatePattern('^[0-9A-Fa-f]{40}$')]
    [string]$ExpectedRootThumbprint,

    [Parameter(Mandatory)]
    [ValidatePattern('^[0-9A-Fa-f]{40}$')]
    [string]$ExpectedLeafThumbprint,

    [string]$InstallFolder,
    [ValidateSet("auto", "cuda", "cpu")]
    [string]$ComputeMode = "auto",
    [string]$LogPath,
    [switch]$Silent,
    [switch]$AuthorizeTrustEnrollment
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-File([string]$Path, [string]$Description) {
    if (-not [System.IO.Path]::IsPathFullyQualified($Path)) {
        throw "$Description path must be absolute: $Path"
    }
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Description was not found: $Path"
    }
    (Resolve-Path -LiteralPath $Path).Path
}

function Assert-Hash([string]$Path, [string]$Expected, [string]$Description) {
    $actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
    if ($actual -cne $Expected.ToUpperInvariant()) {
        throw "$Description SHA-256 does not match the independently approved value"
    }
}

$identity = [System.Security.Principal.WindowsIdentity]::GetCurrent()
$principal = [System.Security.Principal.WindowsPrincipal]::new($identity)
if (-not $principal.IsInRole([System.Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw "Run this script in an elevated PowerShell session authorized by IT/security"
}
if (-not $AuthorizeTrustEnrollment) {
    throw "Trust enrollment is a separate security boundary. Re-run only after approval with -AuthorizeTrustEnrollment"
}

$msi = Resolve-File $MsiPath "Pilot MSI"
$rootCer = Resolve-File $RootCertificatePath "Pilot root public certificate"
$leafCer = Resolve-File $LeafCertificatePath "Pilot leaf public certificate"
$trustScript = Resolve-File (Join-Path $PSScriptRoot "Install-InternalPilotTrust.ps1") "Trust enrollment script"

if ([System.IO.Path]::GetExtension($msi) -cne ".msi") {
    throw "Pilot package must be an MSI"
}
Assert-Hash $msi $ExpectedMsiSha256 "Pilot MSI"
Assert-Hash $rootCer $ExpectedRootCertificateSha256 "Pilot root certificate"
Assert-Hash $leafCer $ExpectedLeafCertificateSha256 "Pilot leaf certificate"

$root = [System.Security.Cryptography.X509Certificates.X509Certificate2]::new(
    [System.IO.File]::ReadAllBytes($rootCer)
)
$leaf = [System.Security.Cryptography.X509Certificates.X509Certificate2]::new(
    [System.IO.File]::ReadAllBytes($leafCer)
)
try {
    if ($root.HasPrivateKey -or $leaf.HasPrivateKey) {
        throw "Pilot inputs must contain public certificates only"
    }
    if ($root.Thumbprint -cne $ExpectedRootThumbprint.ToUpperInvariant() -or
        $leaf.Thumbprint -cne $ExpectedLeafThumbprint.ToUpperInvariant()) {
        throw "Pilot certificate thumbprint does not match the independently approved value"
    }
} finally {
    $root.Dispose()
    $leaf.Dispose()
}

# Validation precedes the explicit, authorized LocalMachine enrollment.
& $trustScript -RootCertificatePath $rootCer -LeafCertificatePath $leafCer -ValidateOnly | Out-Null
& $trustScript -RootCertificatePath $rootCer -LeafCertificatePath $leafCer | Out-Null

$signature = Get-AuthenticodeSignature -LiteralPath $msi
if ($signature.Status -ne [System.Management.Automation.SignatureStatus]::Valid -or
    $null -eq $signature.SignerCertificate -or
    $signature.SignerCertificate.Thumbprint -cne $ExpectedLeafThumbprint.ToUpperInvariant()) {
    throw "Pilot MSI Authenticode signer is not the approved enrolled leaf"
}
if ($null -eq $signature.TimeStamperCertificate) {
    throw "Pilot MSI signature is missing its RFC 3161 timestamp"
}

if ([string]::IsNullOrWhiteSpace($LogPath)) {
    $logDirectory = Join-Path $env:ProgramData "Cuajone PPE Monitor\installer-logs"
    if (-not (Test-Path -LiteralPath $logDirectory)) {
        $logParent = Split-Path -Parent $logDirectory
        if (-not (Test-Path -LiteralPath $logParent -PathType Container)) {
            New-Item -ItemType Directory -Path $logParent | Out-Null
        }
        New-Item -ItemType Directory -Path $logDirectory | Out-Null
    }
    $LogPath = Join-Path $logDirectory (
        "install-{0}.log" -f [DateTime]::UtcNow.ToString("yyyyMMdd-HHmmss")
    )
} else {
    if (-not [System.IO.Path]::IsPathFullyQualified($LogPath)) {
        throw "LogPath must be absolute"
    }
    $logParent = Split-Path -Parent $LogPath
    if (-not (Test-Path -LiteralPath $logParent -PathType Container)) {
        throw "LogPath parent does not exist: $logParent"
    }
}
$resolvedLogPath = [System.IO.Path]::GetFullPath($LogPath)

$arguments = "/i `"$msi`" /norestart /L*V `"$resolvedLogPath`""
$arguments += " COMPUTE_MODE=$ComputeMode"
if (-not [string]::IsNullOrWhiteSpace($InstallFolder)) {
    if (-not [System.IO.Path]::IsPathFullyQualified($InstallFolder)) {
        throw "InstallFolder must be an absolute path"
    }
    $resolvedInstallFolder = [System.IO.Path]::GetFullPath($InstallFolder)
    $arguments += " INSTALLFOLDER=`"$resolvedInstallFolder`""
}
if ($Silent) {
    $arguments += " /qn"
}

$process = Start-Process -FilePath "$env:SystemRoot\System32\msiexec.exe" `
    -ArgumentList $arguments -Wait -PassThru
if ($process.ExitCode -notin @(0, 1641, 3010)) {
    throw "MSI installation failed with exit code $($process.ExitCode); see $resolvedLogPath"
}

[pscustomobject]@{
    Msi = $msi
    MsiSha256 = $ExpectedMsiSha256.ToLowerInvariant()
    SignerThumbprintVerified = $true
    TrustEnrollmentAuthorized = $true
    InstallFolder = if ([string]::IsNullOrWhiteSpace($InstallFolder)) {
        "C:\Program Files\Cuajone PPE Monitor"
    } else {
        $resolvedInstallFolder
    }
    ComputeMode = $ComputeMode
    Log = $resolvedLogPath
    ExitCode = $process.ExitCode
    RebootRequired = $process.ExitCode -in @(1641, 3010)
}
