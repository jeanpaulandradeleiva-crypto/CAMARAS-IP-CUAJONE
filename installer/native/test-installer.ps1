# SPDX-License-Identifier: AGPL-3.0-only

[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$InstallerPath,
    [string]$AcceptanceRoot = "D:\DevTools\CuajoneNative\installer\acceptance",
    [ValidateSet("NotSigned", "Valid")]
    [string]$ExpectedSignatureStatus = "NotSigned"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Assert-DDrivePath([string]$Path, [string]$Description) {
    $fullPath = [System.IO.Path]::GetFullPath($Path)
    if (-not $fullPath.StartsWith("D:\", [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "$Description must remain on D: $fullPath"
    }
}

function Assert-NoForbiddenFiles([string]$Root) {
    $patterns = @(
        '(^|[\\/])\.env($|\.)',
        '\.(pt|engine|onnx|csv|xlsx|xls|pkl|pickle|weights)$',
        '(^|[\\/])(datasets?|weights?|\.atl|caches?|__pycache__)([\\/]|$)',
        '\.(cpp|cxx|cc|h|hpp|lib|pdb|obj|py|pyc)$'
    )
    $violations = foreach ($file in Get-ChildItem -LiteralPath $Root -Recurse -File) {
        $relative = [System.IO.Path]::GetRelativePath($Root, $file.FullName)
        foreach ($pattern in $patterns) {
            if ($relative -match $pattern) {
                $relative
                break
            }
        }
    }
    if ($violations) {
        throw "Forbidden files were installed: $($violations -join ', ')"
    }
}

function Get-SidValue([System.Security.Principal.IdentityReference]$Identity) {
    try {
        return $Identity.Translate(
            [System.Security.Principal.SecurityIdentifier]
        ).Value
    }
    catch {
        throw "Could not resolve ACL identity '$($Identity.Value)' to a SID"
    }
}

function Get-EffectiveRightsForSid(
    [System.Security.AccessControl.FileSystemSecurity]$Acl,
    [string]$Sid
) {
    [long]$allowed = 0
    [long]$denied = 0
    foreach ($rule in $Acl.Access) {
        if ((Get-SidValue $rule.IdentityReference) -ne $Sid) {
            continue
        }
        [long]$rights = $rule.FileSystemRights
        if ($rule.AccessControlType -eq [System.Security.AccessControl.AccessControlType]::Allow) {
            $allowed = $allowed -bor $rights
        }
        else {
            $denied = $denied -bor $rights
        }
    }
    [System.Security.AccessControl.FileSystemRights]($allowed -band (-bnot $denied))
}

function Assert-RequiredRights(
    [System.Security.AccessControl.FileSystemSecurity]$Acl,
    [string]$Sid,
    [System.Security.AccessControl.FileSystemRights]$Required,
    [string]$Description
) {
    $effective = Get-EffectiveRightsForSid $Acl $Sid
    if (($effective -band $Required) -ne $Required) {
        throw "$Description lacks $Required for SID $Sid; effective rights: $effective"
    }
}

function Get-AclEvidence([string]$Path) {
    $acl = Get-Acl -LiteralPath $Path
    [ordered]@{
        path = $Path
        owner = $acl.Owner
        protected = $acl.AreAccessRulesProtected
        entries = @($acl.Access | ForEach-Object {
            [ordered]@{
                identity = $_.IdentityReference.Value
                sid = Get-SidValue $_.IdentityReference
                rights = $_.FileSystemRights.ToString()
                type = $_.AccessControlType.ToString()
                inherited = $_.IsInherited
                inheritanceFlags = $_.InheritanceFlags.ToString()
                propagationFlags = $_.PropagationFlags.ToString()
            }
        })
    }
}

$installer = (Resolve-Path -LiteralPath $InstallerPath).Path
Assert-DDrivePath $installer "Installer"
$installerSignature = Get-AuthenticodeSignature -LiteralPath $installer
if ($installerSignature.Status.ToString() -cne $ExpectedSignatureStatus) {
    throw "Installer signature mismatch: expected $ExpectedSignatureStatus, got $($installerSignature.Status)"
}
Assert-DDrivePath $AcceptanceRoot "Acceptance root"
if (-not (Test-Path -LiteralPath $AcceptanceRoot)) {
    $parent = Split-Path -Parent $AcceptanceRoot
    if (-not (Test-Path -LiteralPath $parent -PathType Container)) {
        throw "Acceptance parent does not exist: $parent"
    }
    New-Item -ItemType Directory -Path $AcceptanceRoot | Out-Null
}

$runName = "run-{0}" -f [DateTime]::UtcNow.ToString("yyyyMMdd-HHmmss")
$runRoot = Join-Path $AcceptanceRoot $runName
$installDir = Join-Path $runRoot "install"
New-Item -ItemType Directory -Path $runRoot | Out-Null
$tempDir = New-Item -ItemType Directory -Path (Join-Path $runRoot "temp")
$installLog = Join-Path $runRoot "install.log"
$uninstallLog = Join-Path $runRoot "uninstall.log"
$originalTemp = $env:TEMP
$originalTmp = $env:TMP
$env:TEMP = $tempDir.FullName
$env:TMP = $tempDir.FullName
$cFreeBefore = (Get-PSDrive -Name C).Free

$install = Start-Process -FilePath $installer -ArgumentList @(
    '/VERYSILENT',
    '/SUPPRESSMSGBOXES',
    '/NORESTART',
    '/SP-',
    "/DIR=$installDir",
    "/LOG=$installLog"
) -Wait -PassThru
if ($install.ExitCode -ne 0) {
    throw "Silent installation failed with exit code $($install.ExitCode); see $installLog"
}

$executable = Join-Path $installDir "bin\cuajone_native.exe"
if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
    throw "Installed executable was not found: $executable"
}
Assert-NoForbiddenFiles $installDir

$requiredInstalledFiles = @(
    "licenses\AGPL-3.0.txt",
    "licenses\FFmpeg-dependencies\AOM-LICENSE.txt",
    "licenses\FFmpeg-dependencies\libvpx-LICENSE.txt",
    "licenses\FFmpeg-dependencies\OpenH264-LICENSE.txt",
    "docs\LICENSES.md",
    "docs\SECURITY.md",
    "docs\SOURCE-OFFER.txt",
    "docs\FFMPEG-SOURCE.md",
    "docs\THIRD_PARTY_NOTICES.md",
    "manifest\build-metadata.json",
    "manifest\SHA256SUMS.txt"
)
foreach ($relativePath in $requiredInstalledFiles) {
    $requiredPath = Join-Path $installDir $relativePath
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "Required license/source-offer file was not installed: $relativePath"
    }
}
$sourceOffer = Get-Content -LiteralPath (Join-Path $installDir "docs\SOURCE-OFFER.txt") -Raw
$buildMetadata = Get-Content -LiteralPath (Join-Path $installDir "manifest\build-metadata.json") -Raw |
    ConvertFrom-Json
if ($sourceOffer -notmatch [regex]::Escape("https://github.com/jeanpaulandradeleiva-crypto/CAMARAS-IP-CUAJONE") -or
    $sourceOffer -notmatch ("Application version:\s+" + [regex]::Escape($buildMetadata.appVersion)) -or
    $sourceOffer -notmatch ("Source revision/correspondence:\s+" + [regex]::Escape($buildMetadata.sourceRevision))) {
    throw "Installed source offer does not identify the repository, version, and source revision"
}
$installedManifest = Get-Content -LiteralPath (Join-Path $installDir "manifest\SHA256SUMS.txt")
foreach ($manifestEntry in @(
    "licenses/AGPL-3.0.txt",
    "docs/SOURCE-OFFER.txt",
    "docs/FFMPEG-SOURCE.md"
)) {
    if (-not ($installedManifest -match ("  " + [regex]::Escape($manifestEntry) + "$"))) {
        throw "Installed manifest does not contain $manifestEntry"
    }
}

$executableSignature = Get-AuthenticodeSignature -LiteralPath $executable
if ($executableSignature.Status.ToString() -cne $ExpectedSignatureStatus) {
    throw "Project executable signature mismatch: expected $ExpectedSignatureStatus, got $($executableSignature.Status)"
}

$mutableDirectories = @(
    (Join-Path $installDir "runtime\models"),
    (Join-Path $installDir "runtime\config"),
    (Join-Path $installDir "runtime\output"),
    (Join-Path $installDir "runtime\logs")
)
foreach ($directory in $mutableDirectories) {
    if (-not (Test-Path -LiteralPath $directory -PathType Container)) {
        throw "Mutable runtime directory was not created: $directory"
    }
}

$originalPath = $env:PATH
$env:PATH = "$(Join-Path $installDir 'bin');$env:SystemRoot\System32;$env:SystemRoot"
try {
    $helpOutput = & $executable --help 2>&1
    $helpExitCode = $LASTEXITCODE
}
finally {
    $env:PATH = $originalPath
}
if ($helpExitCode -ne 0 -or ($helpOutput -join "`n") -notmatch 'Cuajone native TensorRT PPE and fall analytics') {
    throw "Installed loader/help acceptance failed with exit code $helpExitCode"
}
$helpOutput | Set-Content -LiteralPath (Join-Path $runRoot "help-output.txt") -Encoding UTF8

$binDirectory = Join-Path $installDir "bin"
$binAcl = Get-Acl -LiteralPath $binDirectory
$executableAcl = Get-Acl -LiteralPath $executable
$systemSid = "S-1-5-18"
$administratorsSid = "S-1-5-32-544"
$usersSid = "S-1-5-32-545"
$restrictedSids = [ordered]@{
    Users = $usersSid
    AuthenticatedUsers = "S-1-5-11"
    Everyone = "S-1-1-0"
}
$dangerousRights = (
    [System.Security.AccessControl.FileSystemRights]::Write -bor
    [System.Security.AccessControl.FileSystemRights]::Delete -bor
    [System.Security.AccessControl.FileSystemRights]::ChangePermissions -bor
    [System.Security.AccessControl.FileSystemRights]::TakeOwnership
)
foreach ($target in @(
    [pscustomobject]@{ Path = $binDirectory; Acl = $binAcl },
    [pscustomobject]@{ Path = $executable; Acl = $executableAcl }
)) {
    foreach ($principal in $restrictedSids.GetEnumerator()) {
        $effective = Get-EffectiveRightsForSid $target.Acl $principal.Value
        if (($effective -band $dangerousRights) -ne 0) {
            throw "$($principal.Key) can write or modify '$($target.Path)': $effective"
        }
    }
    Assert-RequiredRights $target.Acl $systemSid `
        ([System.Security.AccessControl.FileSystemRights]::FullControl) `
        "SYSTEM on $($target.Path)"
    Assert-RequiredRights $target.Acl $administratorsSid `
        ([System.Security.AccessControl.FileSystemRights]::FullControl) `
        "Administrators on $($target.Path)"
    Assert-RequiredRights $target.Acl $usersSid `
        ([System.Security.AccessControl.FileSystemRights]::ReadAndExecute) `
        "Users on $($target.Path)"
}

foreach ($directory in $mutableDirectories) {
    $acl = Get-Acl -LiteralPath $directory
    Assert-RequiredRights $acl $usersSid `
        ([System.Security.AccessControl.FileSystemRights]::Modify) `
        "Users on mutable runtime directory $directory"
}
$aclEvidence = @(
    (Get-AclEvidence $binDirectory)
    (Get-AclEvidence $executable)
    @($mutableDirectories | ForEach-Object { Get-AclEvidence $_ })
)

$retainedMarker = Join-Path $installDir "runtime\logs\acceptance-retained.txt"
"Created by loader/help acceptance. Safe to delete only after explicit approval." |
    Set-Content -LiteralPath $retainedMarker -Encoding UTF8

$uninstaller = Get-ChildItem -LiteralPath $installDir -Filter "unins*.exe" -File | Select-Object -First 1
if (-not $uninstaller) {
    throw "Uninstaller was not created"
}
$uninstallerSignature = Get-AuthenticodeSignature -LiteralPath $uninstaller.FullName
if ($uninstallerSignature.Status.ToString() -cne $ExpectedSignatureStatus) {
    throw "Uninstaller signature mismatch: expected $ExpectedSignatureStatus, got $($uninstallerSignature.Status)"
}
$uninstall = Start-Process -FilePath $uninstaller.FullName -ArgumentList @(
    '/VERYSILENT',
    '/SUPPRESSMSGBOXES',
    '/NORESTART',
    "/LOG=$uninstallLog"
) -Wait -PassThru
if ($uninstall.ExitCode -ne 0) {
    throw "Silent uninstall failed with exit code $($uninstall.ExitCode); see $uninstallLog"
}

$binRemoved = -not (Test-Path -LiteralPath (Join-Path $installDir "bin"))
$runtimeRetained = @($mutableDirectories | Where-Object {
    -not (Test-Path -LiteralPath $_ -PathType Container)
}).Count -eq 0
$markerRetained = Test-Path -LiteralPath $retainedMarker -PathType Leaf
if (-not $binRemoved -or -not $runtimeRetained -or -not $markerRetained) {
    throw "Uninstall retention failed: binRemoved=$binRemoved runtimeRetained=$runtimeRetained markerRetained=$markerRetained"
}
$cFreeAfter = (Get-PSDrive -Name C).Free
$env:TEMP = $originalTemp
$env:TMP = $originalTmp

$result = [ordered]@{
    installer = $installer
    installerSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $installer).Hash.ToLowerInvariant()
    installerSizeBytes = (Get-Item -LiteralPath $installer).Length
    installerSignatureStatus = $installerSignature.Status.ToString()
    executableSignatureStatus = $executableSignature.Status.ToString()
    uninstallerSignatureStatus = $uninstallerSignature.Status.ToString()
    installerFileVersion = (Get-Item -LiteralPath $installer).VersionInfo.FileVersion.TrimEnd()
    installerProductVersion = (Get-Item -LiteralPath $installer).VersionInfo.ProductVersion.TrimEnd()
    installerOriginalFilename = (Get-Item -LiteralPath $installer).VersionInfo.OriginalFilename.TrimEnd()
    installationMode = "per-machine"
    installExitCode = $install.ExitCode
    helpExitCode = $helpExitCode
    helpLoadedWithRestrictedPath = $true
    forbiddenPatternScan = "passed"
    requiredLicenseAndSourceOfferFiles = "passed"
    manifestLicenseAndSourceOfferEntries = "passed"
    binWriteAccessForRestrictedPrincipals = $false
    aclEvidence = $aclEvidence
    uninstallExitCode = $uninstall.ExitCode
    immutableBinRemoved = $binRemoved
    mutableDirectoriesRetained = $runtimeRetained
    retainedMarker = $retainedMarker
    acceptanceRoot = $runRoot
    redirectedTemp = $tempDir.FullName
    cObservedDeltaBytes = $cFreeAfter - $cFreeBefore
}
$resultPath = Join-Path $runRoot "acceptance-result.json"
$result | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $resultPath -Encoding UTF8
[pscustomobject]$result
