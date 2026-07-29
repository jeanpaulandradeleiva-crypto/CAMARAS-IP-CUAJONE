# SPDX-License-Identifier: AGPL-3.0-only

[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$InstallerPath,

    [Parameter(Mandatory)]
    [string]$StageDir,

    [string]$AcceptanceRoot = "D:\DevTools\CuajoneNative\installer\msi-verification",
    [string]$WixToolRoot = "D:\DevTools\CuajoneNative\wix",
    [string]$SignToolPath = $env:CUAJONE_SIGNTOOL_PATH,

    [ValidateSet("NotSigned", "Signed")]
    [string]$ExpectedSignatureStatus = "NotSigned",

    [switch]$AllowInternalPilotTrust,
    [string]$CertificateThumbprint = $env:CUAJONE_CERTIFICATE_SHA1,
    [string]$PilotRootCertificatePath = $env:CUAJONE_PILOT_ROOT_CER
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$upgradeCode = "88A886C2-8F6D-4669-B6FB-7DFC1E7B0397"
$signatureVerifier = Join-Path $PSScriptRoot "sign-release.ps1"
$payloadPolicy = Join-Path $PSScriptRoot "payload-policy.ps1"

function Assert-File([string]$Path, [string]$Description) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Description was not found: $Path"
    }
}

function Assert-Directory([string]$Path, [string]$Description) {
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        throw "$Description was not found: $Path"
    }
}

function Assert-DDrivePath([string]$Path, [string]$Description) {
    $fullPath = [System.IO.Path]::GetFullPath($Path)
    if (-not $fullPath.StartsWith("D:\", [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "$Description must remain on D: $fullPath"
    }
}

function Get-FileHashWithRetry([string]$Path) {
    for ($attempt = 0; $attempt -lt 600; $attempt++) {
        try {
            return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash
        } catch [System.IO.IOException] {
            Start-Sleep -Milliseconds 100
        } catch [System.UnauthorizedAccessException] {
            Start-Sleep -Milliseconds 100
        }
    }
    throw "Timed out waiting for administrative extraction to release: $Path"
}

function Get-MsiRows([object]$Database, [string]$Query, [int]$ColumnCount) {
    $view = $Database.OpenView($Query)
    try {
        [void]$view.Execute()
        $rows = [System.Collections.Generic.List[object]]::new()
        while ($true) {
            $record = $view.Fetch()
            if ($null -eq $record) {
                break
            }
            try {
                $values = for ($index = 1; $index -le $ColumnCount; $index++) {
                    $record.StringData($index)
                }
                $rows.Add([pscustomobject]@{ Columns = [string[]]@($values) })
            } finally {
                [void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($record)
            }
        }
        return @($rows)
    } finally {
        [void]$view.Close()
        [void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($view)
    }
}

function ConvertTo-PropertyMap([object[]]$Rows) {
    $result = @{}
    foreach ($row in $Rows) {
        $result[$row.Columns[0]] = $row.Columns[1]
    }
    $result
}

$installer = (Resolve-Path -LiteralPath $InstallerPath).Path
$stage = (Resolve-Path -LiteralPath $StageDir).Path
$wix = Join-Path $WixToolRoot "wix.exe"
Assert-DDrivePath $installer "MSI"
Assert-DDrivePath $stage "Stage"
Assert-DDrivePath $AcceptanceRoot "Acceptance root"
Assert-DDrivePath $WixToolRoot "WiX tool root"
Assert-File $installer "MSI"
Assert-Directory $stage "Stage"
Assert-File $wix "WiX CLI"
Assert-File $signatureVerifier "Signature verifier"
Assert-File $payloadPolicy "Installer payload policy"
. $payloadPolicy

$stageMetadataPath = Join-Path $stage "build-metadata.json"
Assert-File $stageMetadataPath "Staged build metadata"
$stageMetadata = Get-Content -LiteralPath $stageMetadataPath -Raw | ConvertFrom-Json
if ($stageMetadata.stagingProvenanceVersion -ne 1 -or $stageMetadata.sourceProvenance.Count -lt 1) {
    throw "Staged build metadata does not contain supported source provenance"
}
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")).Path
if ([System.IO.Path]::GetFullPath($stageMetadata.sourceRoots.repository) -cne $projectRoot) {
    throw "Staged provenance does not identify the current repository root"
}

$classifiedStagePaths = [System.Collections.Generic.HashSet[string]]::new(
    [System.StringComparer]::OrdinalIgnoreCase
)
$sourceFilesCompared = 0
foreach ($entry in $stageMetadata.sourceProvenance) {
    if ($entry.sourceScope -notin @("repository", "build", "tool") -or
        [string]::IsNullOrWhiteSpace($entry.sourcePath) -or
        [string]::IsNullOrWhiteSpace($entry.sourceRelativePath) -or
        [string]::IsNullOrWhiteSpace($entry.stagedRelativePath) -or
        $entry.sourceSha256 -notmatch '^[0-9A-Fa-f]{64}$' -or
        $entry.stagedSha256 -notmatch '^[0-9A-Fa-f]{64}$' -or
        $entry.bytePreserved -ne $true) {
        throw "Invalid source provenance entry in staged build metadata"
    }
    if ([System.IO.Path]::IsPathRooted($entry.stagedRelativePath) -or
        $entry.stagedRelativePath -split '[\\/]' -contains '..') {
        throw "Unsafe staged provenance path: $($entry.stagedRelativePath)"
    }
    if ([System.IO.Path]::IsPathRooted($entry.sourceRelativePath) -or
        $entry.sourceRelativePath -split '[\\/]' -contains '..') {
        throw "Unsafe source provenance path: $($entry.sourceRelativePath)"
    }
    if (-not $classifiedStagePaths.Add($entry.stagedRelativePath)) {
        throw "Duplicate staged provenance path: $($entry.stagedRelativePath)"
    }

    $scopeRoot = $stageMetadata.sourceRoots.PSObject.Properties[$entry.sourceScope].Value
    if ([string]::IsNullOrWhiteSpace($scopeRoot)) {
        throw "Source provenance scope has no declared root: $($entry.sourceScope)"
    }
    $declaredSourcePath = [System.IO.Path]::GetFullPath(
        (Join-Path $scopeRoot $entry.sourceRelativePath)
    )
    if ($declaredSourcePath -cne [System.IO.Path]::GetFullPath($entry.sourcePath)) {
        throw "Source provenance path does not match its scope-relative path: $($entry.sourcePath)"
    }
    Assert-File $entry.sourcePath "Current provenance source"
    $stagedInput = Join-Path $stage $entry.stagedRelativePath
    Assert-File $stagedInput "Provenance staged input"
    $currentSourceHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $entry.sourcePath).Hash.ToLowerInvariant()
    $currentStageHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $stagedInput).Hash.ToLowerInvariant()
    if ($currentSourceHash -cne $entry.sourceSha256.ToLowerInvariant()) {
        throw "Source changed after staging: $($entry.sourcePath)"
    }
    if ($currentStageHash -cne $entry.stagedSha256.ToLowerInvariant() -or
        $currentStageHash -cne $currentSourceHash) {
        throw "Stage no longer corresponds to source: $($entry.stagedRelativePath)"
    }
    $sourceFilesCompared++
}

$expectedRepositoryMappings = [ordered]@{
    "docs/README.md" = "installer/native/README.md"
    "docs/INSTALACION_WINDOWS.md" = "INSTALACION_WINDOWS.md"
    "docs/PROJECT-README.md" = "README.md"
    "docs/SECURITY.md" = "SECURITY.md"
    "docs/LICENSES.md" = "LICENSES.md"
    "docs/FFMPEG-SOURCE.md" = "installer/native/FFMPEG-SOURCE.md"
    "docs/THIRD_PARTY_NOTICES.md" = "installer/native/THIRD_PARTY_NOTICES.md"
    "licenses/AGPL-3.0.txt" = "LICENSE"
}
foreach ($mapping in $expectedRepositoryMappings.GetEnumerator()) {
    $matches = @($stageMetadata.sourceProvenance | Where-Object {
        $_.sourceScope -ceq "repository" -and
        $_.stagedRelativePath.Replace('\', '/') -ceq $mapping.Key -and
        $_.sourceRelativePath.Replace('\', '/') -ceq $mapping.Value
    })
    if ($matches.Count -ne 1) {
        throw "Required repository-to-stage mapping is missing or ambiguous: $($mapping.Value) -> $($mapping.Key)"
    }
}

foreach ($generatedPath in $stageMetadata.generatedStagePaths) {
    if ([string]::IsNullOrWhiteSpace($generatedPath) -or
        [System.IO.Path]::IsPathRooted($generatedPath) -or
        $generatedPath -split '[\\/]' -contains '..' -or
        -not $classifiedStagePaths.Add($generatedPath)) {
        throw "Invalid or duplicate generated stage path: $generatedPath"
    }
    Assert-File (Join-Path $stage $generatedPath) "Declared generated stage file"
}
foreach ($stagedFile in Get-ChildItem -LiteralPath $stage -Recurse -File) {
    $relative = [System.IO.Path]::GetRelativePath($stage, $stagedFile.FullName).Replace('\', '/')
    if (-not $classifiedStagePaths.Contains($relative)) {
        throw "Staged file has no copied-source or generated provenance: $relative"
    }
}
if ($classifiedStagePaths.Count -ne @(Get-ChildItem -LiteralPath $stage -Recurse -File).Count) {
    throw "Stage provenance classification does not exactly match the staged file set"
}

if ($ExpectedSignatureStatus -eq "Signed") {
    Assert-File $SignToolPath "Microsoft signtool"
    $signatureParameters = @{
        FilePath = $installer
        SignToolPath = $SignToolPath
        VerifyOnly = $true
    }
    if ($AllowInternalPilotTrust) {
        $signatureParameters.AllowInternalPilotTrust = $true
        $signatureParameters.CertificateThumbprint = $CertificateThumbprint
        $signatureParameters.PilotRootCertificatePath = $PilotRootCertificatePath
    }
    & $signatureVerifier @signatureParameters | Out-Null
} else {
    $signature = Get-AuthenticodeSignature -LiteralPath $installer
    if ($signature.Status -ne [System.Management.Automation.SignatureStatus]::NotSigned) {
        throw "Expected an unsigned MSI, got $($signature.Status)"
    }
}

if (-not (Test-Path -LiteralPath $AcceptanceRoot)) {
    $parent = Split-Path -Parent $AcceptanceRoot
    Assert-Directory $parent "Acceptance root parent"
    New-Item -ItemType Directory -Path $AcceptanceRoot | Out-Null
}
$runName = "run-{0}" -f [DateTime]::UtcNow.ToString("yyyyMMdd-HHmmss")
$runRoot = Join-Path $AcceptanceRoot $runName
$tempDir = Join-Path $runRoot "temp"
$validateDir = Join-Path $runRoot "validate"
$imageRoot = Join-Path $runRoot "administrative-image"
$extractApp = Join-Path $imageRoot "PFiles64\Cuajone PPE Monitor"
New-Item -ItemType Directory -Path $runRoot | Out-Null
New-Item -ItemType Directory -Path $tempDir | Out-Null
New-Item -ItemType Directory -Path $validateDir | Out-Null
New-Item -ItemType Directory -Path $imageRoot | Out-Null

$originalTemp = $env:TEMP
$originalTmp = $env:TMP
$env:TEMP = $tempDir
$env:TMP = $tempDir
try {
    Push-Location $WixToolRoot
    try {
        $validationOutput = @(& $wix msi validate -intermediateFolder $validateDir $installer 2>&1)
        $validationExitCode = $LASTEXITCODE
    } finally {
        Pop-Location
    }
    $validationOutput | Set-Content -LiteralPath (Join-Path $runRoot "wix-msi-validation.log") -Encoding UTF8
    if ($validationExitCode -ne 0) {
        throw "WiX MSI validation failed with exit code $validationExitCode"
    }

    $windowsInstaller = New-Object -ComObject WindowsInstaller.Installer
    $database = $windowsInstaller.OpenDatabase($installer, 0)
    try {
        $tableRows = Get-MsiRows $database 'SELECT `Name` FROM `_Tables`' 1
        $tables = @($tableRows | ForEach-Object { $_.Columns[0] })
        $requiredTables = @(
            "Component", "Directory", "Feature", "FeatureComponents", "File",
            "InstallExecuteSequence", "Media", "Property", "Shortcut", "Upgrade"
        )
        foreach ($table in $requiredTables) {
            if ($tables -notcontains $table) {
                throw "Required MSI table is missing: $table"
            }
        }

        $properties = ConvertTo-PropertyMap (Get-MsiRows $database 'SELECT `Property`, `Value` FROM `Property`' 2)
        foreach ($requiredProperty in @("ProductCode", "ProductName", "ProductVersion", "Manufacturer", "UpgradeCode")) {
            if ([string]::IsNullOrWhiteSpace($properties[$requiredProperty])) {
                throw "Required MSI property is missing: $requiredProperty"
            }
        }
        if ($properties.ProductName -cne "Cuajone PPE Monitor" -or
            $properties.Manufacturer -cne "Cuajone PPE Monitor Project" -or
            $properties.UpgradeCode -cne "{$upgradeCode}" -and $properties.UpgradeCode -cne $upgradeCode) {
            throw "MSI product identity does not match the approved package identity"
        }
        if ($properties.ALLUSERS -cne "1") {
            throw "MSI is not declared per-machine through ALLUSERS=1"
        }
        $secureProperties = @($properties.SecureCustomProperties -split ';')
        if ($secureProperties -notcontains "INSTALLFOLDER") {
            throw "INSTALLFOLDER is not a secure public MSI property"
        }
        if ($properties.ARPPRODUCTICON -cne "ProductIcon") {
            throw "MSI is missing conventional Add/Remove Programs icon metadata"
        }

        $directoryRows = Get-MsiRows $database 'SELECT `Directory`, `Directory_Parent`, `DefaultDir` FROM `Directory`' 3
        $directoryMap = @{}
        foreach ($row in $directoryRows) {
            $directoryMap[$row.Columns[0]] = [pscustomobject]@{
                Parent = $row.Columns[1]
                DefaultDir = $row.Columns[2]
            }
        }
        if (-not $directoryMap.ContainsKey("INSTALLFOLDER") -or
            $directoryMap.INSTALLFOLDER.Parent -cne "ProgramFiles64Folder" -or
            $directoryMap.INSTALLFOLDER.DefaultDir -notmatch '(^|\|)Cuajone PPE Monitor$') {
            throw "INSTALLFOLDER does not default to ProgramFiles64Folder\Cuajone PPE Monitor"
        }
        if (-not $directoryMap.ContainsKey("APPLICATIONDATAFOLDER") -or
            $directoryMap.APPLICATIONDATAFOLDER.Parent -cne "CommonAppDataFolder") {
            throw "Mutable application data is not rooted under CommonAppDataFolder"
        }
        foreach ($directoryId in @("MODELSFOLDER", "CONFIGFOLDER", "OUTPUTFOLDER", "LOGSFOLDER")) {
            if (-not $directoryMap.ContainsKey($directoryId) -or
                $directoryMap[$directoryId].Parent -cne "RUNTIMEFOLDER") {
                throw "Mutable runtime directory is missing from ProgramData: $directoryId"
            }
        }

        $upgradeRows = Get-MsiRows $database 'SELECT `UpgradeCode`, `VersionMin`, `VersionMax`, `Attributes`, `ActionProperty` FROM `Upgrade`' 5
        if (@($upgradeRows | Where-Object { $_.Columns[0] -in @($upgradeCode, "{$upgradeCode}") }).Count -lt 2) {
            throw "MajorUpgrade rows for upgrade detection and downgrade rejection are missing"
        }
        $launchConditions = Get-MsiRows $database 'SELECT `Condition`, `Description` FROM `LaunchCondition`' 2
        if (@($launchConditions | Where-Object { $_.Columns[0] -match 'WIX_DOWNGRADE_DETECTED' }).Count -ne 1) {
            throw "Downgrade rejection launch condition is missing"
        }

        $sequenceRows = Get-MsiRows $database 'SELECT `Action`, `Condition`, `Sequence` FROM `InstallExecuteSequence`' 3
        $sequenceActions = @($sequenceRows | ForEach-Object { $_.Columns[0] })
        foreach ($action in @("InstallFiles", "RemoveFiles", "RegisterProduct", "PublishProduct", "RemoveExistingProducts")) {
            if ($sequenceActions -notcontains $action) {
                throw "MSI repair/upgrade/uninstall sequence action is missing: $action"
            }
        }

        $shortcutRows = Get-MsiRows $database 'SELECT `Shortcut`, `Directory_`, `Name`, `Target`, `Arguments` FROM `Shortcut`' 5
        if (@($shortcutRows | Where-Object { $_.Columns[1] -ceq "ProgramMenuAppFolder" }).Count -lt 2) {
            throw "Expected Start-menu shortcuts are missing"
        }
        if (@($shortcutRows | Where-Object {
            $_.Columns[0] -ceq "CommandHelpShortcut" -and $_.Columns[4] -match '--help'
        }).Count -ne 1) {
            throw "Command Help shortcut does not declare --help"
        }

        $componentRows = Get-MsiRows $database 'SELECT `Component`, `ComponentId`, `Directory_`, `Attributes`, `KeyPath` FROM `Component`' 5
        if ($componentRows.Count -lt 1 -or @($componentRows | Where-Object {
            ([int]$_.Columns[3] -band 256) -eq 0
        }).Count -ne 0) {
            throw "Every package component must be a deterministic x64 component"
        }
        if (@($componentRows | Where-Object {
            [string]::IsNullOrWhiteSpace($_.Columns[1])
        }).Count -ne 0) {
            throw "A package component is missing its stable GUID"
        }

        $featureComponentRows = Get-MsiRows $database 'SELECT `Feature_`, `Component_` FROM `FeatureComponents`' 2
        if ($featureComponentRows.Count -ne $componentRows.Count) {
            throw "Not every component belongs to the main repairable feature"
        }
        if ($tables -notcontains "Wix4SecureObject") {
            throw "Least-privilege runtime ACL declarations are missing"
        }
        $secureObjectRows = Get-MsiRows $database 'SELECT `SecureObject`, `Table`, `User`, `Permission`, `Component_` FROM `Wix4SecureObject`' 5
        $expectedSecureObjects = @("MODELSFOLDER", "CONFIGFOLDER", "OUTPUTFOLDER", "LOGSFOLDER")
        if ($secureObjectRows.Count -ne $expectedSecureObjects.Count -or
            @($secureObjectRows | Where-Object {
                $_.Columns[0] -notin $expectedSecureObjects -or
                $_.Columns[1] -cne "CreateFolder" -or
                $_.Columns[2] -cne "S-1-5-32-545"
            }).Count -ne 0) {
            throw "Runtime ACL rows do not grant the approved Users SID on the four data directories"
        }
        $customActionRows = Get-MsiRows $database 'SELECT `Action` FROM `CustomAction`' 1
        $approvedCustomActions = @(
            "Wix4SchedSecureObjects_X64", "Wix4SchedSecureObjectsRollback_X64",
            "Wix4ExecSecureObjects_X64", "Wix4ExecSecureObjectsRollback_X64"
        )
        if ($customActionRows.Count -ne $approvedCustomActions.Count -or
            @($customActionRows | Where-Object {
                $_.Columns[0] -notin $approvedCustomActions
            }).Count -ne 0) {
            throw "MSI contains a custom action outside WiX declarative ACL support"
        }
        foreach ($forbiddenTable in @("Certificate", "ServiceInstall")) {
            if ($tables -contains $forbiddenTable) {
                throw "Unexpected security-sensitive MSI table is present: $forbiddenTable"
            }
        }

        $summary = $database.SummaryInformation(0)
        try {
            $template = $summary.Property(7)
            if ($template -notmatch '^x64;') {
                throw "MSI summary template is not x64: $template"
            }
        } finally {
            [void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($summary)
        }
    } finally {
        [void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($database)
        [void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($windowsInstaller)
    }

    $adminLog = Join-Path $runRoot "administrative-extraction.log"
    $adminArguments = @(
        "/a", $installer,
        "/qn",
        "/norestart",
        "TARGETDIR=$imageRoot",
        "/L*V", $adminLog
    )
    & "$env:SystemRoot\System32\msiexec.exe" @adminArguments
    $adminExitCode = $LASTEXITCODE
    if ($adminExitCode -ne 0) {
        throw "MSI administrative extraction failed with exit code $adminExitCode; see $adminLog"
    }
    for ($attempt = 0; $attempt -lt 100 -and
        -not (Test-Path -LiteralPath $extractApp -PathType Container); $attempt++) {
        Start-Sleep -Milliseconds 100
    }
    Assert-Directory $extractApp "Administrative application image"
    $expectedPayloadCount = @(Get-ChildItem -LiteralPath $stage -Recurse -File).Count
    for ($attempt = 0; $attempt -lt 600; $attempt++) {
        $currentPayloadCount = @(Get-ChildItem -LiteralPath $extractApp -Recurse -File).Count
        if ($currentPayloadCount -ge $expectedPayloadCount) {
            break
        }
        Start-Sleep -Milliseconds 100
    }
    Assert-NoForbiddenPayloadFiles $extractApp "administrative image"

    $payloadCount = 0
    foreach ($stagedFile in Get-ChildItem -LiteralPath $stage -Recurse -File) {
        $relative = [System.IO.Path]::GetRelativePath($stage, $stagedFile.FullName)
        $extractedFile = Join-Path $extractApp $relative
        Assert-File $extractedFile "Extracted payload file"
        $stageHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $stagedFile.FullName).Hash
        $extractedHash = Get-FileHashWithRetry $extractedFile
        if ($extractedHash -cne $stageHash) {
            throw "Administrative extraction changed payload bytes: $relative"
        }
        $payloadCount++
    }
    $extractedPayloadCount = @(Get-ChildItem -LiteralPath $extractApp -Recurse -File).Count
    if ($extractedPayloadCount -ne $payloadCount) {
        throw "Administrative image file count mismatch: stage=$payloadCount extracted=$extractedPayloadCount"
    }

    $metadataPath = Join-Path $extractApp "build-metadata.json"
    $metadata = Get-Content -LiteralPath $metadataPath -Raw | ConvertFrom-Json
    foreach ($binary in $metadata.stagedBinaries) {
        $extractedBinary = Join-Path $extractApp "bin\$($binary.name)"
        $extractedHash = Get-FileHashWithRetry $extractedBinary
        if ($extractedHash.ToLowerInvariant() -cne $binary.sha256.ToLowerInvariant()) {
            throw "Extracted binary differs from the approved source bytes: $($binary.name)"
        }
    }

    $executable = Join-Path $extractApp "bin\cuajone_native.exe"
    if ($ExpectedSignatureStatus -eq "Signed") {
        $executableSignatureParameters = @{
            FilePath = $executable
            SignToolPath = $SignToolPath
            VerifyOnly = $true
        }
        if ($AllowInternalPilotTrust) {
            $executableSignatureParameters.AllowInternalPilotTrust = $true
            $executableSignatureParameters.CertificateThumbprint = $CertificateThumbprint
            $executableSignatureParameters.PilotRootCertificatePath = $PilotRootCertificatePath
        }
        & $signatureVerifier @executableSignatureParameters | Out-Null
    } elseif ((Get-AuthenticodeSignature -LiteralPath $executable).Status -ne
        [System.Management.Automation.SignatureStatus]::NotSigned) {
        throw "Expected the extracted owned executable to be unsigned"
    }

    $originalPath = $env:PATH
    $env:PATH = "$(Join-Path $extractApp 'bin');$env:SystemRoot\System32;$env:SystemRoot"
    try {
        $helpOutput = & $executable --help 2>&1
        $helpExitCode = $LASTEXITCODE
    } finally {
        $env:PATH = $originalPath
    }
    if ($helpExitCode -ne 0 -or ($helpOutput -join "`n") -notmatch 'Cuajone native TensorRT PPE and fall analytics') {
        throw "Extracted loader/help acceptance failed with exit code $helpExitCode"
    }
    $helpOutput | Set-Content -LiteralPath (Join-Path $runRoot "help-output.txt") -Encoding UTF8
} finally {
    $env:TEMP = $originalTemp
    $env:TMP = $originalTmp
}

$result = [ordered]@{
    installer = $installer
    installerSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $installer).Hash.ToLowerInvariant()
    installerSizeBytes = (Get-Item -LiteralPath $installer).Length
    signatureExpectation = $ExpectedSignatureStatus
    msiDatabaseOpen = "passed"
    wixMsiValidation = "passed"
    productIdentity = "passed"
    installFolderProperty = "public-secure-configurable"
    installFolderDefault = "ProgramFiles64Folder\Cuajone PPE Monitor"
    dataFolder = "CommonAppDataFolder\Cuajone PPE Monitor\runtime"
    packageScope = "per-machine-x64"
    majorUpgradeAndDowngradeBlock = "passed"
    repairAndUninstallTables = "passed"
    startMenuAndArpMetadata = "passed"
    deterministicComponents = "passed"
    sourceCorrespondence = "passed"
    sourceFilesCompared = $sourceFilesCompared
    administrativeExtractionExitCode = $adminExitCode
    payloadFilesCompared = $payloadCount
    thirdPartyBytePreservation = "passed"
    helpExitCode = $helpExitCode
    liveInstallPerformed = $false
    localMachineCertificateStoresModified = $false
    acceptanceRoot = $runRoot
}
$resultPath = Join-Path $runRoot "verification-result.json"
$result | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $resultPath -Encoding UTF8
[pscustomobject]$result
