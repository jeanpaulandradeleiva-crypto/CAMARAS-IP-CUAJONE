# SPDX-License-Identifier: AGPL-3.0-only

[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$InstallerPath,

    [Parameter(Mandatory)]
    [string]$StageDir,

    [string]$AcceptanceRoot,
    [string]$WixToolRoot,
    [string]$SignToolPath = $env:CUAJONE_SIGNTOOL_PATH,

    [ValidateSet("NotSigned", "Signed")]
    [string]$ExpectedSignatureStatus = "NotSigned",

    [switch]$AllowInternalPilotTrust,
    [string]$CertificateThumbprint = $env:CUAJONE_CERTIFICATE_SHA1,
    [string]$PilotRootCertificatePath = $env:CUAJONE_PILOT_ROOT_CER,

    [switch]$FastPreview
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$upgradeCode = "88A886C2-8F6D-4669-B6FB-7DFC1E7B0397"
$signatureVerifier = Join-Path $PSScriptRoot "sign-release.ps1"
$payloadPolicy = Join-Path $PSScriptRoot "payload-policy.ps1"
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")).Path
$toolRoot = Join-Path $projectRoot ".tools\native"
if ([string]::IsNullOrWhiteSpace($AcceptanceRoot)) { $AcceptanceRoot = Join-Path $toolRoot "installer\msi-verification" }
if ([string]::IsNullOrWhiteSpace($WixToolRoot)) { $WixToolRoot = Join-Path $toolRoot "wix" }

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

function Assert-ToolRootPath([string]$Path, [string]$Description) {
    $fullPath = [System.IO.Path]::GetFullPath($Path)
    $fullToolRoot = [System.IO.Path]::GetFullPath($toolRoot).TrimEnd('\')
    if ($fullPath -cne $fullToolRoot -and -not $fullPath.StartsWith("$fullToolRoot\", [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "$Description must remain under the repository-local tool root: $fullPath"
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

function Get-PeSubsystem([string]$Path) {
    Assert-File $Path "PE file"
    $stream = [System.IO.File]::OpenRead($Path)
    $reader = [System.IO.BinaryReader]::new($stream)
    try {
        if ($stream.Length -lt 64 -or $reader.ReadUInt16() -ne 0x5A4D) {
            throw "File does not contain a valid DOS header: $Path"
        }
        $stream.Position = 0x3C
        $peOffset = [long]$reader.ReadUInt32()
        if ($peOffset -lt 64 -or $peOffset + 94 -gt $stream.Length) {
            throw "File contains an invalid PE header offset: $Path"
        }
        $stream.Position = $peOffset
        if ($reader.ReadUInt32() -ne 0x00004550) {
            throw "File does not contain a valid PE signature: $Path"
        }
        $stream.Position = $peOffset + 24
        $magic = $reader.ReadUInt16()
        if ($magic -notin @(0x010B, 0x020B)) {
            throw "File contains an unsupported PE optional header: $Path"
        }
        $stream.Position = $peOffset + 24 + 68
        $reader.ReadUInt16()
    } finally {
        $reader.Dispose()
        $stream.Dispose()
    }
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
Assert-ToolRootPath $installer "MSI"
Assert-ToolRootPath $stage "Stage"
Assert-ToolRootPath $AcceptanceRoot "Acceptance root"
Assert-ToolRootPath $WixToolRoot "WiX tool root"
Assert-File $installer "MSI"
Assert-Directory $stage "Stage"
Assert-File $wix "WiX CLI"
Assert-File $signatureVerifier "Signature verifier"
Assert-File $payloadPolicy "Installer payload policy"
. $payloadPolicy

$sidecarPath = "$installer.sha256"
Assert-File $sidecarPath "MSI SHA-256 sidecar"
$expectedActiveNames = @(
    (Split-Path -Leaf $installer)
    (Split-Path -Leaf $sidecarPath)
)
$activeCandidates = @(Get-ChildItem -LiteralPath (Split-Path -Parent $installer) -File | Where-Object {
    $_.Name -like 'NexoAIVision-*-x64*.msi' -or
    $_.Name -like 'NexoAIVision-*-x64*.msi.sha256' -or
    $_.Name -like 'CuajonePPEMonitor-*-x64*.msi' -or
    $_.Name -like 'CuajonePPEMonitor-*-x64*.msi.sha256'
})
if ($activeCandidates.Count -ne 2 -or @($activeCandidates.Name | Where-Object {
        $_ -notin $expectedActiveNames
    }).Count -ne 0) {
    throw "Active output does not contain exactly the selected MSI and its sidecar"
}
$sidecarLine = (Get-Content -LiteralPath $sidecarPath -Raw).Trim()
$expectedSidecar = "{0}  {1}" -f (
    (Get-FileHash -Algorithm SHA256 -LiteralPath $installer).Hash.ToLowerInvariant()
), (Split-Path -Leaf $installer)
if ($sidecarLine -cne $expectedSidecar) {
    throw "MSI SHA-256 sidecar does not match the selected installer"
}

$stageMetadataPath = Join-Path $stage "build-metadata.json"
Assert-File $stageMetadataPath "Staged build metadata"
$stagedLauncher = Join-Path $stage "bin\NexoAIVisionLauncher.exe"
$stagedRuntime = Join-Path $stage "bin\NexoAIVision.exe"
Assert-File $stagedLauncher "Staged launcher executable"
Assert-File $stagedRuntime "Staged runtime executable"
$stageMetadata = Get-Content -LiteralPath $stageMetadataPath -Raw | ConvertFrom-Json
if ($stageMetadata.stagingProvenanceVersion -ne 1 -or $stageMetadata.sourceProvenance.Count -lt 1) {
    throw "Staged build metadata does not contain supported source provenance"
}
if ($stageMetadata.onnxRuntime.version -cne "1.25.0" -or
    $stageMetadata.onnxRuntime.assetSha256 -cne "125c9fe408f41b9ae1ad7138dac5ebb19a85e65438d1e368d21b50e6abb32f4e" -or
    $stageMetadata.onnxRuntime.executionProvider -cne "CPUExecutionProvider + CUDAExecutionProvider") {
    throw "Staged build metadata does not pin the approved ONNX Runtime GPU package"
}
$approvedOnnxRuntimeCore = Join-Path $stageMetadata.onnxRuntime.root "lib\onnxruntime.dll"
Assert-File $approvedOnnxRuntimeCore "Approved ONNX Runtime GPU core"
$stagedOnnxRuntimeCore = Join-Path $stage "bin\onnxruntime.dll"
Assert-File $stagedOnnxRuntimeCore "Staged ONNX Runtime core"
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $stagedOnnxRuntimeCore).Hash -cne
    (Get-FileHash -Algorithm SHA256 -LiteralPath $approvedOnnxRuntimeCore).Hash) {
    throw "Staged onnxruntime.dll does not match the approved GPU package core"
}
if ((Split-Path -Leaf $stageMetadata.launcherExecutable) -cne "NexoAIVisionLauncher.exe" -or
    $stageMetadata.launcherExecutableSha256 -cne (Get-FileHash -Algorithm SHA256 -LiteralPath $stagedLauncher).Hash.ToLowerInvariant() -or
    (Split-Path -Leaf $stageMetadata.releaseExecutable) -cne "NexoAIVision.exe" -or
    $stageMetadata.releaseExecutableSha256 -cne (Get-FileHash -Algorithm SHA256 -LiteralPath $stagedRuntime).Hash.ToLowerInvariant()) {
    throw "Staged build metadata does not identify both owned executables and their hashes"
}
foreach ($ownedExecutableName in @("NexoAIVisionLauncher.exe", "NexoAIVision.exe")) {
    if (@($stageMetadata.stagedBinaries | Where-Object {
        $_.name -ceq $ownedExecutableName
    }).Count -ne 1) {
        throw "Staged build metadata does not contain exactly one owned executable root: $ownedExecutableName"
    }
}
if ($stageMetadata.runtimeSecurityPolicy.hardwareProbeSchemaVersion -ne 2 -or
    $stageMetadata.runtimeSecurityPolicy.minimumCudaDriverApiVersion -ne 12090 -or
    $stageMetadata.runtimeSecurityPolicy.minimumTensorRtComputeCapability -cne "7.5" -or
    $stageMetadata.runtimeSecurityPolicy.onnxManifestSchemaVersion -ne 1 -or
    $stageMetadata.runtimeSecurityPolicy.onnxExternalDataAllowed -ne $false -or
    $stageMetadata.runtimeSecurityPolicy.onnxCustomOperatorsAllowed -ne $false -or
    $stageMetadata.runtimeSecurityPolicy.maximumOnnxModelBytes -ne 268435456 -or
    $stageMetadata.runtimeSecurityPolicy.maximumTensorRtEngineBytes -ne 1073741824 -or
    $stageMetadata.runtimeSecurityPolicy.maximumImageDimension -ne 4096 -or
    $stageMetadata.runtimeSecurityPolicy.maximumOutputElements -ne 16777216 -or
    $stageMetadata.runtimeSecurityPolicy.maximumTensorBytes -ne 268435456) {
    throw "Staged build metadata does not record the enforced runtime security policy"
}
$sbomPath = Join-Path $stage "docs\sbom.spdx.json"
Assert-File $sbomPath "SPDX SBOM"
$sbom = Get-Content -LiteralPath $sbomPath -Raw | ConvertFrom-Json
if ($sbom.spdxVersion -cne "SPDX-2.3" -or $sbom.dataLicense -cne "CC0-1.0" -or
    @($sbom.packages | Where-Object {
        $_.name -ceq "onnxruntime.dll" -and $_.versionInfo -ceq "1.25.0" -and
        $_.licenseDeclared -ceq "MIT"
    }).Count -ne 1) {
    throw "SPDX SBOM does not identify the approved ONNX Runtime package"
}
if ([System.IO.Path]::GetFullPath($stageMetadata.sourceRoots.repository) -cne $projectRoot) {
    throw "Staged provenance does not identify the current repository root"
}
$fastPreviewProperty = $stageMetadata.PSObject.Properties["fastPreview"]
$stageWasFastPreview = ($null -ne $fastPreviewProperty) -and $fastPreviewProperty.Value -eq $true
if ($FastPreview -and -not $stageWasFastPreview) {
    throw "FastPreview verification requires a stage built with -FastPreview (metadata.fastPreview)"
}
if (-not $FastPreview -and $stageWasFastPreview) {
    throw "Full verification cannot be applied to a fast-preview stage; rebuild without -FastPreview"
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
    if (-not $FastPreview) {
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
$extractApp = Join-Path $imageRoot "PFiles64\NexoAI Vision"
New-Item -ItemType Directory -Path $runRoot | Out-Null
New-Item -ItemType Directory -Path $tempDir | Out-Null
New-Item -ItemType Directory -Path $validateDir | Out-Null
if (-not $FastPreview) {
    New-Item -ItemType Directory -Path $imageRoot | Out-Null
}

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
            "AppSearch", "Binary", "Component", "Control", "ControlCondition", "ControlEvent",
            "CustomAction", "Dialog", "Directory", "Feature", "FeatureComponents", "File",
            "InstallExecuteSequence", "InstallUISequence", "Media", "Property", "RegLocator",
            "RadioButton", "Registry", "Shortcut", "Upgrade", "Wix4SecureObject"
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
        if ($properties.ProductName -cne "NexoAI Vision" -or
            $properties.Manufacturer -cne "NexoAI Vision Project" -or
            $properties.UpgradeCode -cne "{$upgradeCode}" -and $properties.UpgradeCode -cne $upgradeCode) {
            throw "MSI product identity does not match the approved package identity"
        }
        if ($properties.ALLUSERS -cne "1") {
            throw "MSI is not declared per-machine through ALLUSERS=1"
        }
        $secureProperties = @($properties.SecureCustomProperties -split ';')
        foreach ($secureProperty in @("INSTALLFOLDER", "COMPUTE_MODE", "CUDA_READY", "NVIDIA_STATUS")) {
            if ($secureProperties -notcontains $secureProperty) {
                throw "$secureProperty is not a secure public MSI property"
            }
        }
        if ($properties.COMPUTE_MODE -cne "auto" -or $properties.CUDA_READY -cne "0" -or
            $properties.NVIDIA_STATUS -cne "not_probed") {
            throw "Compute properties do not have safe defaults"
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
            $directoryMap.INSTALLFOLDER.DefaultDir -notmatch '(^|\|)NexoAI Vision$') {
            throw "INSTALLFOLDER does not default to ProgramFiles64Folder\NexoAI Vision"
        }
        if (-not $directoryMap.ContainsKey("APPLICATIONDATAFOLDER") -or
            $directoryMap.APPLICATIONDATAFOLDER.Parent -cne "CommonAppDataFolder" -or
            $directoryMap.APPLICATIONDATAFOLDER.DefaultDir -notmatch '(^|\|)NexoAI Vision$') {
            throw "Mutable application data is not rooted under CommonAppDataFolder\NexoAI Vision"
        }
        foreach ($directoryId in @("CONFIGFOLDER", "OUTPUTFOLDER", "LOGSFOLDER")) {
            if (-not $directoryMap.ContainsKey($directoryId) -or
                $directoryMap[$directoryId].Parent -cne "RUNTIMEFOLDER") {
                throw "Mutable runtime directory is missing from ProgramData: $directoryId"
            }
        }
        $appSearchRows = Get-MsiRows $database 'SELECT `Property`, `Signature_` FROM `AppSearch`' 2
        $registrySearchRows = Get-MsiRows $database 'SELECT `Signature_`, `Root`, `Key`, `Name` FROM `RegLocator`' 4
        foreach ($expectedSearch in @(
                [pscustomobject]@{ Property = "PREVIOUS_COMPUTE_MODE"; Key = "SOFTWARE\NexoAI Vision" },
                [pscustomobject]@{ Property = "LEGACY_COMPUTE_MODE"; Key = "SOFTWARE\Cuajone PPE Monitor" }
            )) {
            $appSearch = @($appSearchRows | Where-Object { $_.Columns[0] -ceq $expectedSearch.Property })
            if ($appSearch.Count -ne 1 -or @($registrySearchRows | Where-Object {
                    $_.Columns[0] -ceq $appSearch[0].Columns[1] -and $_.Columns[1] -eq "2" -and
                    $_.Columns[2] -ceq $expectedSearch.Key -and $_.Columns[3] -ceq "ComputeMode"
                }).Count -ne 1) {
                throw "Compute-mode migration search is missing: $($expectedSearch.Key)"
            }
        }
        $secureObjectRows = Get-MsiRows $database 'SELECT `Domain`, `User`, `Component_` FROM `Wix4SecureObject`' 3
        if ($secureObjectRows.Count -ne 4 -or @($secureObjectRows | Where-Object {
            -not [string]::IsNullOrEmpty($_.Columns[0]) -or $_.Columns[1] -cne "Users"
        }).Count -ne 0) {
            throw "Runtime directory permissions do not use the locale-independent WiX Users account"
        }

        $upgradeRows = Get-MsiRows $database 'SELECT `UpgradeCode`, `VersionMin`, `VersionMax`, `Attributes`, `ActionProperty` FROM `Upgrade`' 5
        if (@($upgradeRows | Where-Object { $_.Columns[0] -in @($upgradeCode, "{$upgradeCode}") }).Count -lt 2) {
            throw "MajorUpgrade rows for upgrade detection and downgrade rejection are missing"
        }
        $launchConditions = Get-MsiRows $database 'SELECT `Condition`, `Description` FROM `LaunchCondition`' 2
        if (@($launchConditions | Where-Object { $_.Columns[0] -match 'WIX_DOWNGRADE_DETECTED' }).Count -ne 1) {
            throw "Downgrade rejection launch condition is missing"
        }
        if (@($launchConditions | Where-Object {
            $_.Columns[0] -match 'COMPUTE_MODE' -and $_.Columns[1] -match 'auto, cuda, or cpu'
        }).Count -ne 1) {
            throw "Invalid COMPUTE_MODE values are not rejected"
        }
        if (@($launchConditions | Where-Object {
            $_.Columns[0] -match 'VersionNT|WindowsBuild'
        }).Count -ne 0) {
            throw "MSI contains an unsupported Windows version launch restriction"
        }

        $sequenceRows = Get-MsiRows $database 'SELECT `Action`, `Condition`, `Sequence` FROM `InstallExecuteSequence`' 3
        $sequenceActions = @($sequenceRows | ForEach-Object { $_.Columns[0] })
        foreach ($action in @("InstallFiles", "RemoveFiles", "RegisterProduct", "PublishProduct", "RemoveExistingProducts")) {
            if ($sequenceActions -notcontains $action) {
                throw "MSI repair/upgrade/uninstall sequence action is missing: $action"
            }
        }
        foreach ($action in @("DetectComputeHardware", "BlockUnavailableCuda")) {
            if ($sequenceActions -notcontains $action) {
                throw "Compute gate is missing from InstallExecuteSequence: $action"
            }
        }
        $detectSequence = @($sequenceRows | Where-Object { $_.Columns[0] -ceq "DetectComputeHardware" })
        $blockSequence = @($sequenceRows | Where-Object { $_.Columns[0] -ceq "BlockUnavailableCuda" })
        if ($detectSequence.Count -ne 1 -or $blockSequence.Count -ne 1 -or
            [int]$detectSequence[0].Columns[2] -ge [int]$blockSequence[0].Columns[2] -or
            $blockSequence[0].Columns[1] -notmatch 'COMPUTE_MODE="cuda"' -or
            $blockSequence[0].Columns[1] -notmatch 'CUDA_READY') {
            throw "Forced CUDA is not blocked after the hardware probe"
        }
        $uiSequenceRows = Get-MsiRows $database 'SELECT `Action`, `Condition`, `Sequence` FROM `InstallUISequence`' 3
        if (@($uiSequenceRows | Where-Object { $_.Columns[0] -ceq "DetectComputeHardware" }).Count -ne 1) {
            throw "Interactive compute detection is missing from InstallUISequence"
        }

        $dialogRows = Get-MsiRows $database 'SELECT `Dialog`, `Title` FROM `Dialog`' 2
        if (@($dialogRows | Where-Object {
                $_.Columns[0] -ceq "ComputeDlg" -and $_.Columns[1] -ceq "NexoAI Vision - Compute"
            }).Count -ne 1) {
            throw "Compute selection dialog is missing or has the wrong product title"
        }
        $controlRows = Get-MsiRows $database 'SELECT `Dialog_`, `Control`, `Type`, `Property`, `Text` FROM `Control`' 5
        foreach ($control in @("ComputeModeSelection", "ProbeStatus")) {
            if (@($controlRows | Where-Object {
                $_.Columns[0] -ceq "ComputeDlg" -and $_.Columns[1] -ceq $control
            }).Count -ne 1) {
                throw "Compute dialog control is missing: $control"
            }
        }
        $computeModeControl = @($controlRows | Where-Object {
            $_.Columns[0] -ceq "ComputeDlg" -and $_.Columns[1] -ceq "ComputeModeSelection"
        })
        if ($computeModeControl.Count -ne 1 -or $computeModeControl[0].Columns[2] -cne "RadioButtonGroup" -or
            $computeModeControl[0].Columns[3] -cne "COMPUTE_MODE") {
            throw "Compute mode is not represented by a property-bound radio button group"
        }
        $radioButtonRows = Get-MsiRows $database 'SELECT `Property`, `Value`, `Text` FROM `RadioButton`' 3
        $computeModeRadioRows = @($radioButtonRows | Where-Object { $_.Columns[0] -ceq "COMPUTE_MODE" })
        if ($computeModeRadioRows.Count -ne 3) {
            throw "Compute mode radio button group does not contain exactly three choices"
        }
        foreach ($mode in @("auto", "cuda", "cpu")) {
            if (@($computeModeRadioRows | Where-Object {
                $_.Columns[1] -ceq $mode
            }).Count -ne 1) {
                throw "Compute dialog does not expose radio choice: $mode"
            }
        }

        $shortcutRows = Get-MsiRows $database 'SELECT `Shortcut`, `Directory_`, `Name`, `Component_`, `Target`, `Arguments` FROM `Shortcut`' 6
        if (@($shortcutRows | Where-Object { $_.Columns[1] -ceq "ProgramMenuAppFolder" }).Count -lt 3) {
            throw "Expected Start-menu shortcuts are missing"
        }
        $launcherShortcut = @($shortcutRows | Where-Object {
            $_.Columns[0] -ceq "LauncherShortcut" -and
            ($_.Columns[2] -split '\|')[-1] -ceq "NexoAI Vision" -and
            [string]::IsNullOrEmpty($_.Columns[5])
        })
        if ($launcherShortcut.Count -ne 1) {
            throw "Primary graphical launcher shortcut is missing or has unexpected arguments"
        }
        $helpShortcut = @($shortcutRows | Where-Object {
            $_.Columns[0] -ceq "CommandHelpShortcut" -and $_.Columns[5] -match '--help'
        })
        if ($helpShortcut.Count -ne 1) {
            throw "Command Help shortcut does not declare --help"
        }
        $readmeShortcut = @($shortcutRows | Where-Object {
            $_.Columns[0] -ceq "ReadmeShortcut" -and
            ($_.Columns[2] -split '\|')[-1] -ceq "NexoAI Vision - README"
        })
        if ($readmeShortcut.Count -ne 1) {
            throw "README shortcut is missing"
        }

        $fileRows = Get-MsiRows $database 'SELECT `File`, `Component_`, `FileName` FROM `File`' 3
        foreach ($requiredExecutable in @("NexoAIVisionLauncher.exe", "NexoAIVision.exe")) {
            if (@($fileRows | Where-Object {
                ($_.Columns[2] -split '\|')[-1] -ceq $requiredExecutable
            }).Count -ne 1) {
                throw "Owned executable payload is missing or duplicated: $requiredExecutable"
            }
        }
        $launcherFile = @($fileRows | Where-Object { $_.Columns[0] -ceq "LauncherExecutable" })
        $runtimeFile = @($fileRows | Where-Object { $_.Columns[0] -ceq "RuntimeExecutable" })
        $readmeFile = @($fileRows | Where-Object { $_.Columns[0] -ceq "DeploymentReadme" })
        if ($launcherFile.Count -ne 1 -or $launcherShortcut[0].Columns[3] -cne $launcherFile[0].Columns[1]) {
            throw "Primary shortcut is not bound to the launcher executable component"
        }
        if ($runtimeFile.Count -ne 1 -or $helpShortcut[0].Columns[3] -cne $runtimeFile[0].Columns[1]) {
            throw "Command Help shortcut is not bound to the runtime executable component"
        }
        if ($readmeFile.Count -ne 1 -or $readmeShortcut[0].Columns[3] -cne $readmeFile[0].Columns[1]) {
            throw "README shortcut is not bound to the deployment README component"
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

        $featureRows = Get-MsiRows $database 'SELECT `Feature`, `Title` FROM `Feature`' 2
        if (@($featureRows | Where-Object {
                $_.Columns[0] -ceq "MainFeature" -and $_.Columns[1] -ceq "NexoAI Vision"
            }).Count -ne 1) {
            throw "Main feature does not use the NexoAI Vision title"
        }
        $featureComponentRows = Get-MsiRows $database 'SELECT `Feature_`, `Component_` FROM `FeatureComponents`' 2
        if ($featureComponentRows.Count -ne $componentRows.Count) {
            throw "Not every component belongs to the main repairable feature"
        }
        if ($tables -notcontains "Wix4SecureObject") {
            throw "Least-privilege runtime ACL declarations are missing"
        }
        $secureObjectRows = Get-MsiRows $database 'SELECT `SecureObject`, `Table`, `User`, `Permission`, `Component_` FROM `Wix4SecureObject`' 5
        $expectedSecureObjects = @("CONFIGFOLDER", "OUTPUTFOLDER", "LOGSFOLDER")
        if ($secureObjectRows.Count -ne $expectedSecureObjects.Count -or
            @($secureObjectRows | Where-Object {
                $_.Columns[0] -notin $expectedSecureObjects -or
                $_.Columns[1] -cne "CreateFolder" -or
                $_.Columns[2] -cne "Users"
            }).Count -ne 0) {
            throw "Runtime ACL rows do not grant the locale-independent Users account on the three mutable data directories"
        }
        $customActionRows = Get-MsiRows $database 'SELECT `Action`, `Type`, `Source`, `Target` FROM `CustomAction`' 4
        $approvedCustomActions = @(
            "Wix4SchedSecureObjects_X64", "Wix4SchedSecureObjectsRollback_X64",
            "Wix4ExecSecureObjects_X64", "Wix4ExecSecureObjectsRollback_X64",
            "DetectComputeHardware", "BlockUnavailableCuda",
            "RestoreLegacyComputeMode", "RestoreNexoAIComputeMode"
        )
        if ($customActionRows.Count -ne $approvedCustomActions.Count -or
            @($customActionRows | Where-Object {
                $_.Columns[0] -notin $approvedCustomActions
            }).Count -ne 0) {
            throw "MSI contains an unapproved custom action"
        }
        $probeAction = @($customActionRows | Where-Object { $_.Columns[0] -ceq "DetectComputeHardware" })
        if ($probeAction.Count -ne 1 -or $probeAction[0].Columns[2] -cne "HardwareProbeCA" -or
            $probeAction[0].Columns[3] -cne "DetectComputeHardware") {
            throw "Hardware probe custom action does not reference the approved embedded DLL/export"
        }
        $binaryRows = Get-MsiRows $database 'SELECT `Name` FROM `Binary`' 1
        if (@($binaryRows | Where-Object { $_.Columns[0] -ceq "HardwareProbeCA" }).Count -ne 1) {
            throw "Hardware probe custom action binary is not embedded exactly once"
        }

        $registryRows = Get-MsiRows $database 'SELECT `Registry`, `Root`, `Key`, `Name`, `Value`, `Component_` FROM `Registry`' 6
        if (@($registryRows | Where-Object {
            $_.Columns[1] -eq "2" -and $_.Columns[2] -ceq "SOFTWARE\NexoAI Vision" -and
            $_.Columns[3] -ceq "ComputeMode" -and $_.Columns[4] -ceq "[COMPUTE_MODE]" -and
            $_.Columns[5] -ceq "RuntimeComputeMode"
        }).Count -ne 1) {
            throw "Selected compute mode is not persisted in 64-bit HKLM"
        }
        foreach ($forbiddenTable in @("Certificate", "ServiceInstall")) {
            if ($tables -contains $forbiddenTable) {
                throw "Unexpected security-sensitive MSI table is present: $forbiddenTable"
            }
        }

        $summary = $database.SummaryInformation(0)
        try {
            if ($summary.Property(3) -cne "NexoAI Vision $($stageMetadata.appVersion) x64") {
                throw "MSI summary description does not use the NexoAI Vision product name"
            }
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

    if (-not $FastPreview) {
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

        $launcher = Join-Path $extractApp "bin\NexoAIVisionLauncher.exe"
        $executable = Join-Path $extractApp "bin\NexoAIVision.exe"
        Assert-File $launcher "Extracted launcher executable"
        Assert-File $executable "Extracted runtime executable"
        foreach ($ownedExecutable in @($launcher, $executable)) {
            if ($ExpectedSignatureStatus -eq "Signed") {
                $executableSignatureParameters = @{
                    FilePath = $ownedExecutable
                    SignToolPath = $SignToolPath
                    VerifyOnly = $true
                }
                if ($AllowInternalPilotTrust) {
                    $executableSignatureParameters.AllowInternalPilotTrust = $true
                    $executableSignatureParameters.CertificateThumbprint = $CertificateThumbprint
                    $executableSignatureParameters.PilotRootCertificatePath = $PilotRootCertificatePath
                }
                & $signatureVerifier @executableSignatureParameters | Out-Null
            } elseif ((Get-AuthenticodeSignature -LiteralPath $ownedExecutable).Status -ne
                [System.Management.Automation.SignatureStatus]::NotSigned) {
                throw "Expected the extracted owned executable to be unsigned: $ownedExecutable"
            }
        }
        $launcherSubsystem = Get-PeSubsystem $launcher
        $runtimeSubsystem = Get-PeSubsystem $executable
        if ($launcherSubsystem -ne 2) {
            throw "Launcher is not a Windows GUI subsystem executable: $launcherSubsystem"
        }
        if ($runtimeSubsystem -ne 3) {
            throw "Runtime is not a Windows console subsystem executable: $runtimeSubsystem"
        }

        $originalPath = $env:PATH
        $env:PATH = "$(Join-Path $extractApp 'bin');$env:SystemRoot\System32;$env:SystemRoot"
        try {
            $helpOutput = & $executable --help 2>&1
            $helpExitCode = $LASTEXITCODE
        } finally {
            $env:PATH = $originalPath
        }
        if ($helpExitCode -ne 0 -or ($helpOutput -join "`n") -notmatch 'NexoAI Vision PPE and fall analytics') {
            throw "Extracted loader/help acceptance failed with exit code $helpExitCode"
        }
        $helpOutput | Set-Content -LiteralPath (Join-Path $runRoot "help-output.txt") -Encoding UTF8

        $originalPath = $env:PATH
        $env:PATH = "$(Join-Path $extractApp 'bin');$env:SystemRoot\System32;$env:SystemRoot"
        try {
            $probeOutput = & $executable --hardware-probe-json 2>&1
            $probeExitCode = $LASTEXITCODE
        } finally {
            $env:PATH = $originalPath
        }
        if ($probeExitCode -notin @(0, 10, 11, 12, 13)) {
            throw "Extracted hardware probe returned an undocumented exit code: $probeExitCode"
        }
        $probeJson = ($probeOutput -join "`n") | ConvertFrom-Json
        if ($probeJson.schema_version -ne 2 -or
            $probeJson.minimum_driver_version -ne 12090 -or
            $probeJson.status -notin @("cuda_ready", "no_nvidia_adapter", "driver_unavailable", "driver_too_old", "probe_error") -or
            $probeJson.cuda_ready -ne ($probeJson.status -ceq "cuda_ready") -or
            $probeJson.driver_was_loaded -ne $false) {
            throw "Extracted hardware probe JSON violates the stable schema"
        }
        $probeOutput | Set-Content -LiteralPath (Join-Path $runRoot "hardware-probe.json") -Encoding UTF8
    } else {
        $stagedLauncher = Join-Path $stage "bin\NexoAIVisionLauncher.exe"
        $stagedRuntime = Join-Path $stage "bin\NexoAIVision.exe"
        Assert-File $stagedLauncher "Staged launcher executable"
        Assert-File $stagedRuntime "Staged runtime executable"
        foreach ($ownedExecutable in @($stagedLauncher, $stagedRuntime)) {
            if ($ExpectedSignatureStatus -eq "Signed") {
                $executableSignatureParameters = @{
                    FilePath = $ownedExecutable
                    SignToolPath = $SignToolPath
                    VerifyOnly = $true
                }
                if ($AllowInternalPilotTrust) {
                    $executableSignatureParameters.AllowInternalPilotTrust = $true
                    $executableSignatureParameters.CertificateThumbprint = $CertificateThumbprint
                    $executableSignatureParameters.PilotRootCertificatePath = $PilotRootCertificatePath
                }
                & $signatureVerifier @executableSignatureParameters | Out-Null
            } elseif ((Get-AuthenticodeSignature -LiteralPath $ownedExecutable).Status -ne
                [System.Management.Automation.SignatureStatus]::NotSigned) {
                throw "Expected the staged owned executable to be unsigned: $ownedExecutable"
            }
        }
        $launcherSubsystem = Get-PeSubsystem $stagedLauncher
        $runtimeSubsystem = Get-PeSubsystem $stagedRuntime
        if ($launcherSubsystem -ne 2) {
            throw "Staged launcher is not a Windows GUI subsystem executable: $launcherSubsystem"
        }
        if ($runtimeSubsystem -ne 3) {
            throw "Staged runtime is not a Windows console subsystem executable: $runtimeSubsystem"
        }

        $originalPath = $env:PATH
        $env:PATH = "$(Join-Path $stage 'bin');$env:SystemRoot\System32;$env:SystemRoot"
        try {
            $helpOutput = & $stagedRuntime --help 2>&1
            $helpExitCode = $LASTEXITCODE
        } finally {
            $env:PATH = $originalPath
        }
        if ($helpExitCode -ne 0 -or ($helpOutput -join "`n") -notmatch 'NexoAI Vision PPE and fall analytics') {
            throw "Staged loader/help smoke failed with exit code $helpExitCode"
        }
        $helpOutput | Set-Content -LiteralPath (Join-Path $runRoot "help-output.txt") -Encoding UTF8
        Assert-NoForbiddenPayloadFiles $stage "installer stage (FastPreview)"
    }
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
    installFolderDefault = "ProgramFiles64Folder\NexoAI Vision"
    dataFolder = "CommonAppDataFolder\NexoAI Vision\runtime"
    packageScope = "per-machine-x64"
    majorUpgradeAndDowngradeBlock = "passed"
    repairAndUninstallTables = "passed"
    startMenuAndArpMetadata = "passed"
    deterministicComponents = "passed"
    fastPreview = [bool]$FastPreview
    sourceCorrespondence = if ($FastPreview) { "asserted-at-staging-fast-preview" } else { "passed" }
    sourceFilesCompared = $sourceFilesCompared
    administrativeExtractionExitCode = if ($FastPreview) { $null } else { $adminExitCode }
    payloadFilesCompared = if ($FastPreview) { 0 } else { $payloadCount }
    thirdPartyBytePreservation = if ($FastPreview) { "skipped-fast-preview" } else { "passed" }
    launcherSubsystem = "windows-gui"
    runtimeSubsystem = "windows-console"
    helpExitCode = $helpExitCode
    hardwareProbeExitCode = if ($FastPreview) { $null } else { $probeExitCode }
    hardwareProbeStatus = if ($FastPreview) { "not-run-fast-preview" } else { $probeJson.status }
    cudaDriverEagerlyLoaded = if ($FastPreview) { $null } else { $probeJson.driver_was_loaded }
    computeSelectionContract = "auto-cuda-cpu-persisted-and-fail-closed"
    liveInstallPerformed = $false
    localMachineCertificateStoresModified = $false
    acceptanceRoot = $runRoot
}
$resultPath = Join-Path $runRoot "verification-result.json"
$result | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $resultPath -Encoding UTF8
[pscustomobject]$result
