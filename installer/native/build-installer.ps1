# SPDX-License-Identifier: AGPL-3.0-only

[CmdletBinding()]
param(
    [ValidatePattern('^\d+\.\d+\.\d+(-[0-9A-Za-z.-]+)?$')]
    [string]$Version = "0.1.0-internal.4",

    [ValidatePattern('^\d+\.\d+\.\d+\.\d+$')]
    [string]$FileVersion = "0.1.0.4",

    [ValidateSet("Preview", "Release")]
    [string]$BuildMode = "Release",

    [switch]$AllowUnsignedPreview,
    [switch]$SkipModelBundle,
    [switch]$RefreshModelExport,
    [switch]$FastPreview,
    [switch]$StageOnly,

    [string]$ToolRoot,
    [string]$WixToolRoot,
    [string]$WixVersion = "6.0.2",
    [string]$ReleaseExecutable,
    [string]$LauncherExecutable,
    [string]$HardwareProbeCustomAction,
    [string]$StageDir,
    [string]$WixBuildDir,
    [string]$OutputDir,
    [string]$SupersededOutputDir,
    [string]$VerificationRoot,
    [string]$SourceRevision = $env:CUAJONE_SOURCE_REVISION,
    [string]$SourceArchiveUrl = $env:CUAJONE_SOURCE_ARCHIVE_URL,
    [string]$SourceArchiveSha256 = $env:CUAJONE_SOURCE_ARCHIVE_SHA256,
    [string]$FfmpegSourceArchiveUrl = $env:CUAJONE_FFMPEG_SOURCE_ARCHIVE_URL,
    [string]$FfmpegSourceArchiveSha256 = $env:CUAJONE_FFMPEG_SOURCE_ARCHIVE_SHA256,
    [string]$PythonExecutable = $env:CUAJONE_PYTHON_EXECUTABLE,
    [string]$PpeModelPath = $env:CUAJONE_PPE_MODEL_PATH,
    [string]$PoseModelPath = $env:CUAJONE_POSE_MODEL_PATH,
    [string]$PpeEnginePath = $env:CUAJONE_PPE_ENGINE_PATH,
    [string]$PoseEnginePath = $env:CUAJONE_POSE_ENGINE_PATH,
    [string]$OnnxRuntimeGpuRoot = $env:CUAJONE_ONNXRUNTIME_GPU_ROOT,
    [string]$ParityReceiptPath = $env:CUAJONE_PARITY_RECEIPT,
    [string]$SignToolPath = $env:CUAJONE_SIGNTOOL_PATH,
    [string]$SignCommand = $env:CUAJONE_SIGN_COMMAND
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptRoot = $PSScriptRoot
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $scriptRoot "..\..")).Path
$localToolRoot = Join-Path $projectRoot ".tools\native"
if ([string]::IsNullOrWhiteSpace($ToolRoot)) { $ToolRoot = $localToolRoot }
if ([string]::IsNullOrWhiteSpace($WixToolRoot)) { $WixToolRoot = Join-Path $ToolRoot "wix" }
$releaseDirectory = Join-Path $ToolRoot "build\presets\windows-msvc"
if ([string]::IsNullOrWhiteSpace($ReleaseExecutable)) { $ReleaseExecutable = Join-Path $releaseDirectory "cuajone_native.exe" }
if ([string]::IsNullOrWhiteSpace($LauncherExecutable)) { $LauncherExecutable = Join-Path $releaseDirectory "cuajone_launcher.exe" }
if ([string]::IsNullOrWhiteSpace($HardwareProbeCustomAction)) { $HardwareProbeCustomAction = Join-Path $releaseDirectory "CuajoneHardwareProbeCA.dll" }
if ([string]::IsNullOrWhiteSpace($StageDir)) { $StageDir = Join-Path $ToolRoot "installer\stage" }
if ([string]::IsNullOrWhiteSpace($WixBuildDir)) { $WixBuildDir = Join-Path $ToolRoot "installer\wix-build" }
if ([string]::IsNullOrWhiteSpace($OutputDir)) { $OutputDir = Join-Path $ToolRoot "installer\output" }
if ([string]::IsNullOrWhiteSpace($SupersededOutputDir)) { $SupersededOutputDir = Join-Path $ToolRoot "installer\superseded" }
if ([string]::IsNullOrWhiteSpace($VerificationRoot)) { $VerificationRoot = Join-Path $ToolRoot "installer\msi-verification" }
$packageSource = Join-Path $scriptRoot "Package.wxs"
$packageProject = Join-Path $scriptRoot "CuajonePpeMonitor.wixproj"
$iconGenerator = Join-Path $scriptRoot "generate-icon.ps1"
$signatureVerifier = Join-Path $scriptRoot "sign-release.ps1"
$packageVerifier = Join-Path $scriptRoot "test-installer.ps1"
$payloadPolicy = Join-Path $scriptRoot "payload-policy.ps1"
$releaseGates = Join-Path $scriptRoot "release-gates.ps1"
$sourceRepository = "https://github.com/jeanpaulandradeleiva-crypto/CAMARAS-IP-CUAJONE"
$upgradeCode = "88A886C2-8F6D-4669-B6FB-7DFC1E7B0397"
$onnxRuntimeVersion = "1.25.0"
$onnxRuntimeAssetUrl = "https://github.com/microsoft/onnxruntime/releases/download/v1.25.0/onnxruntime-win-x64-1.25.0.zip"
$onnxRuntimeAssetSha256 = "da753f762bf2400e7191ec594086b186a7051d5af8dc886f6e2020c2403df738"

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
    $fullToolRoot = [System.IO.Path]::GetFullPath($localToolRoot).TrimEnd('\')
    if ($fullPath -cne $fullToolRoot -and -not $fullPath.StartsWith("$fullToolRoot\", [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "$Description must remain under the repository-local tool root: $fullPath"
    }
}

function Assert-HttpsUrl([string]$Value, [string]$Description) {
    if ([string]::IsNullOrWhiteSpace($Value) -or $Value -notmatch '^https://') {
        throw "$Description must be an HTTPS URL"
    }
}

function Assert-Sha256([string]$Value, [string]$Description) {
    if ([string]::IsNullOrWhiteSpace($Value) -or $Value -notmatch '^[0-9A-Fa-f]{64}$') {
        throw "$Description must be a 64-character SHA-256 value"
    }
}

function Resolve-FirstExistingPath([string[]]$Candidates, [string]$Description) {
    foreach ($candidate in $Candidates) {
        if ([string]::IsNullOrWhiteSpace($candidate)) {
            continue
        }
        $resolved = $candidate
        if (-not [System.IO.Path]::IsPathRooted($resolved)) {
            $resolved = Join-Path $projectRoot $resolved
        }
        if (Test-Path -LiteralPath $resolved -PathType Leaf) {
            return (Resolve-Path -LiteralPath $resolved).Path
        }
    }
    throw "$Description was not found. Checked: $($Candidates -join ', ')"
}

function Resolve-ModelSource([string]$ConfiguredPath, [string[]]$FallbackCandidates, [string]$Description) {
    if (-not [string]::IsNullOrWhiteSpace($ConfiguredPath)) {
        if (-not [System.IO.Path]::IsPathRooted($ConfiguredPath)) {
            $ConfiguredPath = Join-Path $projectRoot $ConfiguredPath
        }
        Assert-File $ConfiguredPath $Description
        return (Resolve-Path -LiteralPath $ConfiguredPath).Path
    }
    return Resolve-FirstExistingPath $FallbackCandidates $Description
}

function Export-OnnxModel(
    [string]$PythonPath,
    [string]$SourceModel,
    [string]$OutputPath,
    [string]$Task
) {
    Assert-File $PythonPath "Python executable for ONNX export"
    Assert-File $SourceModel "Source model"
    $outputParent = Split-Path -Parent $OutputPath
    Ensure-Directory $outputParent

    $pyCode = "import sys; from ultralytics import YOLO; model = YOLO(r'$SourceModel'); model.export(format='onnx', imgsz=640)"
    $output = @(& $PythonPath -c $pyCode 2>&1)
    if ($LASTEXITCODE -ne 0) {
        throw "ONNX export failed for $SourceModel`n$($output -join [Environment]::NewLine)"
    }

    $generatedOnnx = [System.IO.Path]::ChangeExtension($SourceModel, ".onnx")
    if (-not (Test-Path -LiteralPath $generatedOnnx -PathType Leaf)) {
        throw "Expected ONNX file not found at $generatedOnnx after export"
    }
    Copy-Item -LiteralPath $generatedOnnx -Destination $OutputPath -Force
    Assert-File $OutputPath "Exported ONNX model"
    $sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $OutputPath).Hash.ToLowerInvariant()

    [pscustomobject]@{
        source = (Resolve-Path -LiteralPath $SourceModel).Path
        sourceSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $SourceModel).Hash.ToLowerInvariant()
        task = $Task
        onnx = $OutputPath
        onnxSha256 = $sha256
    }
}

function Write-OnnxManifest(
    [string]$PythonPath,
    [string]$OnnxPath,
    [ValidateSet("ppe", "pose")]
    [string]$Role,
    [string]$SourceModel
) {
    Assert-File $PythonPath "Python executable for ONNX manifest generation"
    Assert-File $OnnxPath "ONNX model"
    Assert-File $SourceModel "Source model"
    $manifestPath = "$OnnxPath.manifest.json"
    $python = @'
import hashlib
import json
import sys

import onnx

model_path, role, source_model = sys.argv[1:]
model = onnx.load(model_path, load_external_data=False)

def tensor_contract(values):
    if len(values) != 1:
        raise RuntimeError("exported ONNX model must expose exactly one input and one output")
    tensor = values[0].type.tensor_type
    if tensor.elem_type != onnx.TensorProto.FLOAT:
        raise RuntimeError("exported ONNX tensors must use float32")
    shape = []
    for dimension in tensor.shape.dim:
        if not dimension.HasField("dim_value") or dimension.dim_value <= 0:
            raise RuntimeError("exported ONNX tensors must use fixed positive dimensions")
        shape.append(dimension.dim_value)
    return {"name": values[0].name, "element_type": "float32", "shape": shape}

with open(model_path, "rb") as stream:
    model_bytes = stream.read()
manifest = {
    "schema_version": 1,
    "artifact_type": "onnx",
    "role": role,
    "model_file": model_path.rsplit("\\", 1)[-1].rsplit("/", 1)[-1],
    "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
    "model_size_bytes": len(model_bytes),
    "external_data": False,
    "custom_operators": False,
    "input": tensor_contract(model.graph.input),
    "output": tensor_contract(model.graph.output),
    "provenance": {
        "source_uri": "urn:cuajone:bundled-model:" + source_model,
        "exporter": "ultralytics-onnx-export",
        "license": "NOASSERTION",
    },
}
print(json.dumps(manifest, separators=(",", ":")))
'@
    $output = @(& $PythonPath -c $python $OnnxPath $Role ([System.IO.Path]::GetFileName($SourceModel)) 2>&1)
    if ($LASTEXITCODE -ne 0) {
        throw "ONNX manifest generation failed for $OnnxPath`n$($output -join [Environment]::NewLine)"
    }
    ($output -join [Environment]::NewLine) | Set-Content -LiteralPath $manifestPath -Encoding utf8 -NoNewline
    Assert-File $manifestPath "ONNX model manifest"
    [pscustomobject]@{
        path = $manifestPath
        sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $manifestPath).Hash.ToLowerInvariant()
    }
}

function Get-StringSha256([string]$Value) {
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($Value)
    $sha256 = [System.Security.Cryptography.SHA256]::Create()
    try {
        return ([System.BitConverter]::ToString($sha256.ComputeHash($bytes))).Replace("-", "").ToLowerInvariant()
    } finally {
        $sha256.Dispose()
    }
}

function Get-UltralyticsVersion([string]$PythonPath) {
    Assert-File $PythonPath "Python executable for ONNX export"
    $output = @(& $PythonPath -c "import ultralytics; print(ultralytics.__version__)" 2>&1)
    if ($LASTEXITCODE -ne 0 -or $output.Count -ne 1 -or [string]::IsNullOrWhiteSpace($output[0])) {
        throw "Could not determine the Ultralytics version for ONNX export`n$($output -join [Environment]::NewLine)"
    }
    return $output[0].Trim()
}

function Get-OrExportOnnxModel(
    [string]$PythonPath,
    [string]$SourceModel,
    [string]$CacheRoot,
    [ValidateSet("ppe", "pose")]
    [string]$Role,
    [string]$Task,
    [switch]$Refresh
) {
    Assert-File $SourceModel "Source model"
    Ensure-Directory $CacheRoot
    $sourceSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $SourceModel).Hash.ToLowerInvariant()
    $exporterVersion = Get-UltralyticsVersion $PythonPath
    $recipe = "role=$Role;task=$Task;imgsz=640;source_sha256=$sourceSha256;ultralytics=$exporterVersion"
    $cacheDirectory = Join-Path $CacheRoot (Get-StringSha256 $recipe)
    $onnxPath = Join-Path $cacheDirectory "$Role.onnx"
    $manifestPath = "$onnxPath.manifest.json"
    $receiptPath = Join-Path $cacheDirectory "export-receipt.json"
    $cacheHit = $false

    if (-not $Refresh -and (Test-Path -LiteralPath $onnxPath -PathType Leaf) -and (Test-Path -LiteralPath $manifestPath -PathType Leaf) -and (Test-Path -LiteralPath $receiptPath -PathType Leaf)) {
        try {
            $receipt = Get-Content -LiteralPath $receiptPath -Raw | ConvertFrom-Json
            $onnxSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $onnxPath).Hash.ToLowerInvariant()
            $manifestSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $manifestPath).Hash.ToLowerInvariant()
            $cacheHit = $receipt.recipe -ceq $recipe `
                -and $receipt.source_sha256 -ceq $sourceSha256 `
                -and $receipt.onnx_sha256 -ceq $onnxSha256 `
                -and $receipt.manifest_sha256 -ceq $manifestSha256
        } catch {
            $cacheHit = $false
        }
    }

    if (-not $cacheHit) {
        Reset-Directory $cacheDirectory
        $exported = Export-OnnxModel $PythonPath $SourceModel $onnxPath $Task
        $manifest = Write-OnnxManifest $PythonPath $exported.onnx $Role $SourceModel
        $receipt = [ordered]@{
            recipe = $recipe
            source_sha256 = $exported.sourceSha256
            onnx_sha256 = $exported.onnxSha256
            manifest_sha256 = $manifest.sha256
        }
        $temporaryReceipt = "$receiptPath.tmp"
        $receipt | ConvertTo-Json -Compress | Set-Content -LiteralPath $temporaryReceipt -Encoding utf8 -NoNewline
        Move-Item -LiteralPath $temporaryReceipt -Destination $receiptPath -Force
    } else {
        $exported = [pscustomobject]@{
            source = (Resolve-Path -LiteralPath $SourceModel).Path
            sourceSha256 = $sourceSha256
            task = $Task
            onnx = $onnxPath
            onnxSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $onnxPath).Hash.ToLowerInvariant()
        }
        $manifest = [pscustomobject]@{
            path = $manifestPath
            sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $manifestPath).Hash.ToLowerInvariant()
        }
    }

    return [pscustomobject]@{
        exported = $exported
        manifest = $manifest
        cacheHit = $cacheHit
    }
}

function Ensure-Directory([string]$Path) {
    Assert-ToolRootPath $Path "Generated directory"
    if (Test-Path -LiteralPath $Path) {
        Assert-Directory $Path "Generated directory"
        return
    }
    New-Item -ItemType Directory -Path $Path -Force | Out-Null
}

function Reset-Directory([string]$Path) {
    Assert-ToolRootPath $Path "Generated directory"
    if (Test-Path -LiteralPath $Path) {
        Remove-Item -LiteralPath $Path -Recurse -Force
    }
    New-Item -ItemType Directory -Path $Path -Force | Out-Null
}

function Ensure-OnnxRuntimeCudaPackage([string]$TargetRoot) {
    $provider = Join-Path $TargetRoot "lib\onnxruntime_providers_cuda.dll"
    if (Test-Path -LiteralPath $provider -PathType Leaf) {
        return
    }
    $archive = Join-Path $ToolRoot "downloads\onnxruntime-win-x64-gpu-1.25.0.zip"
    Ensure-Directory (Split-Path -Parent $archive)
    if (-not (Test-Path -LiteralPath $archive -PathType Leaf)) {
        Invoke-WebRequest -Uri "https://github.com/microsoft/onnxruntime/releases/download/v1.25.0/onnxruntime-win-x64-gpu-1.25.0.zip" -OutFile $archive
    }
    $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $archive).Hash.ToLowerInvariant()
    if ($hash -cne "125c9fe408f41b9ae1ad7138dac5ebb19a85e65438d1e368d21b50e6abb32f4e") {
        throw "ONNX Runtime CUDA archive hash does not match the approved 1.25.0 asset"
    }
    $parent = Split-Path -Parent $TargetRoot
    Expand-Archive -LiteralPath $archive -DestinationPath $parent -Force
    Assert-File $provider "Extracted ONNX Runtime CUDA provider"
}

function Expand-VerifiedNvidiaWheel(
    [string]$Archive,
    [string]$ExpectedSha256,
    [string]$TargetRoot,
    [string]$RequiredDll,
    [string]$Description) {
    Assert-File $Archive $Description
    $actualSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $Archive).Hash.ToLowerInvariant()
    if ($actualSha256 -cne $ExpectedSha256) {
        throw "$Description hash does not match the approved package"
    }
    $marker = Join-Path $TargetRoot $RequiredDll
    if (-not (Test-Path -LiteralPath $marker -PathType Leaf)) {
        Reset-Directory $TargetRoot
        Expand-Archive -LiteralPath $Archive -DestinationPath $TargetRoot -Force
    }
    Assert-File $marker "$Description required DLL"
}

function Get-PeDependencies([string]$Path, [string]$Dumpbin) {
    $output = & $Dumpbin /dependents $Path 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "dumpbin failed for $Path`n$($output -join [Environment]::NewLine)"
    }
    $dependencies = foreach ($line in $output) {
        if ([string]$line -match '^\s+([A-Za-z0-9_.+-]+\.dll)\s*$') {
            $Matches[1]
        }
    }
    @($dependencies | Sort-Object -Unique)
}

function Assert-X64Pe([string]$Path, [string]$Dumpbin) {
    $headers = & $Dumpbin /headers $Path 2>&1
    if ($LASTEXITCODE -ne 0 -or ($headers -join "`n") -notmatch '(?im)^\s*8664 machine \(x64\)') {
        throw "Expected an x64 PE file: $Path"
    }
}

function Find-Dependency([string]$Name, [string[]]$SearchDirectories) {
    foreach ($directory in $SearchDirectories) {
        $candidate = Join-Path $directory $Name
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }
    return $null
}

function Get-StableHex([string]$Value) {
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($Value.ToLowerInvariant())
    $hash = [System.Security.Cryptography.SHA256]::HashData($bytes)
    [System.Convert]::ToHexString($hash).ToLowerInvariant()
}

function Get-DeterministicComponentGuid([string]$RelativePath) {
    $hex = Get-StableHex "$upgradeCode|$RelativePath"
    $bytes = [System.Convert]::FromHexString($hex.Substring(0, 32))
    $bytes[7] = ($bytes[7] -band 0x0F) -bor 0x50
    $bytes[8] = ($bytes[8] -band 0x3F) -bor 0x80
    [System.Guid]::new($bytes).ToString().ToUpperInvariant()
}

function ConvertTo-XmlValue([string]$Value) {
    [System.Security.SecurityElement]::Escape($Value)
}

function Write-PayloadSource {
    param(
        [string]$Root,
        [string]$OutputPath,
        [string]$ComponentGroupName = "PayloadComponents",
        [string]$IncludePrefix = $null,
        [string]$ExcludePrefix = $null,
        [string[]]$IncludeRelativePaths = @(),
        [string[]]$ExcludeRelativePaths = @(),
        [string]$ComponentCondition = $null
    )
    $lines = [System.Collections.Generic.List[string]]::new()
    $lines.Add('<!-- Generated by build-installer.ps1. Do not commit. -->')
    $lines.Add('<Wix xmlns="http://wixtoolset.org/schemas/v4/wxs">')
    $lines.Add('  <Fragment>')
    $lines.Add("    <ComponentGroup Id=`"$ComponentGroupName`" Directory=`"INSTALLFOLDER`">")

    foreach ($file in Get-ChildItem -LiteralPath $Root -Recurse -File | Sort-Object FullName) {
        $relative = [System.IO.Path]::GetRelativePath($Root, $file.FullName).Replace('/', '\')
        if ($IncludePrefix -and -not $relative.StartsWith($IncludePrefix, [StringComparison]::OrdinalIgnoreCase)) {
            continue
        }
        if ($ExcludePrefix -and $relative.StartsWith($ExcludePrefix, [StringComparison]::OrdinalIgnoreCase)) {
            continue
        }
        if ($IncludeRelativePaths.Count -gt 0 -and $relative -notin $IncludeRelativePaths) {
            continue
        }
        if ($relative -in $ExcludeRelativePaths) {
            continue
        }
        $hex = Get-StableHex $relative
        $componentId = "Component_$($hex.Substring(0, 24))"
        $fileId = switch -CaseSensitive ($relative) {
            'bin\cuajone_launcher.exe' { 'LauncherExecutable'; break }
            'bin\cuajone_native.exe' { 'RuntimeExecutable'; break }
            'docs\README.md' { 'DeploymentReadme'; break }
            'NexoAIVision.ico' { 'InstalledProductIcon'; break }
            default { "File_$($hex.Substring(0, 24))" }
        }
        $subdirectory = Split-Path -Parent $relative
        $subdirectoryAttribute = if ([string]::IsNullOrWhiteSpace($subdirectory)) {
            ""
        } else {
            " Subdirectory=`"$(ConvertTo-XmlValue $subdirectory)`""
        }
        $source = ConvertTo-XmlValue $file.FullName
        $guid = Get-DeterministicComponentGuid $relative
        $conditionAttribute = if ([string]::IsNullOrWhiteSpace($ComponentCondition)) {
            ""
        } else {
            " Condition=`"$(ConvertTo-XmlValue $ComponentCondition)`""
        }
        $lines.Add("      <Component Id=`"$componentId`" Guid=`"$guid`"$subdirectoryAttribute$conditionAttribute>")
        if ($fileId -ceq 'LauncherExecutable') {
            $lines.Add("        <File Id=`"$fileId`" Source=`"$source`" KeyPath=`"yes`">")
            $lines.Add('          <Shortcut Id="LauncherShortcut" Directory="ProgramMenuAppFolder" Name="NexoAI Vision" Description="Configure and start NexoAI Vision" WorkingDirectory="INSTALLFOLDER" Advertise="yes" Icon="LauncherIcon.exe" IconIndex="0" />')
            $lines.Add('        </File>')
            $lines.Add('        <RemoveFolder Id="RemoveProgramMenuAppFolder" Directory="ProgramMenuAppFolder" On="uninstall" />')
        } elseif ($fileId -ceq 'RuntimeExecutable') {
            $lines.Add("        <File Id=`"$fileId`" Source=`"$source`" KeyPath=`"yes`">")
            $lines.Add('          <Shortcut Id="CommandHelpShortcut" Directory="ProgramMenuAppFolder" Name="NexoAI Vision - Command Help" Description="Open command-line help" Arguments="--help" WorkingDirectory="INSTALLFOLDER" Advertise="yes" />')
            $lines.Add('        </File>')
        } elseif ($fileId -ceq 'DeploymentReadme') {
            $lines.Add("        <File Id=`"$fileId`" Source=`"$source`" KeyPath=`"yes`">")
            $lines.Add('          <Shortcut Id="ReadmeShortcut" Directory="ProgramMenuAppFolder" Name="NexoAI Vision - README" Description="Open deployment and license documentation" WorkingDirectory="INSTALLFOLDER" Advertise="yes" />')
            $lines.Add('        </File>')
        } else {
            $lines.Add("        <File Id=`"$fileId`" Source=`"$source`" KeyPath=`"yes`" />")
        }
        $lines.Add('      </Component>')
    }

    $lines.Add('    </ComponentGroup>')
    $lines.Add('  </Fragment>')
    $lines.Add('</Wix>')
    $lines | Set-Content -LiteralPath $OutputPath -Encoding UTF8
}

foreach ($path in @($StageDir, $WixBuildDir, $OutputDir, $SupersededOutputDir, $VerificationRoot, $WixToolRoot)) {
    Assert-ToolRootPath $path "Build path"
}
Assert-Directory $ToolRoot "Native tool root"
Assert-Directory $WixToolRoot "WiX tool root"
Assert-File $ReleaseExecutable "Release executable"
Assert-File $LauncherExecutable "Launcher executable"
Assert-File $HardwareProbeCustomAction "Hardware probe custom action"
if ((Split-Path -Leaf $ReleaseExecutable) -cne "cuajone_native.exe") {
    throw "ReleaseExecutable must identify cuajone_native.exe"
}
if ((Split-Path -Leaf $LauncherExecutable) -cne "cuajone_launcher.exe") {
    throw "LauncherExecutable must identify cuajone_launcher.exe"
}
Assert-File $packageSource "WiX package source"
Assert-File $packageProject "WiX project"
Assert-File $iconGenerator "Icon generator"
Assert-File $signatureVerifier "Authenticode signing helper"
Assert-File $packageVerifier "MSI verification helper"
Assert-File $payloadPolicy "Installer payload policy"
Assert-File $releaseGates "Release parity gate"
Assert-File (Join-Path $projectRoot "LICENSE") "Project AGPL license"
. $payloadPolicy
. $releaseGates

$gitHead = (& git -C $projectRoot rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $gitHead -notmatch '^[0-9a-f]{40}$') {
    throw "Could not determine the repository HEAD revision"
}
$gitStatus = @(& git -C $projectRoot status --porcelain=v1 --untracked-files=all)
$isDirty = $gitStatus.Count -gt 0
$isSignedBuild = -not [string]::IsNullOrWhiteSpace($SignCommand)
$isInternalPilotSigning = $env:CUAJONE_ALLOW_INTERNAL_PILOT_TRUST -ceq "1"
$productionParityReceipt = $null
$cabinetCompressionLevel = Get-CabinetCompressionLevel $FastPreview

Assert-FastIterationFlags -FastPreview:$FastPreview -StageOnly:$StageOnly -BuildMode $BuildMode

if ($isInternalPilotSigning -and $BuildMode -ne "Preview") {
    throw "Internal pilot signing is permitted only for Preview builds; Release requires public trust"
}
if ($isInternalPilotSigning -and -not $isSignedBuild) {
    throw "Internal pilot trust was enabled without CUAJONE_SIGN_COMMAND"
}
if ($BuildMode -eq "Release") {
    if (-not $isSignedBuild) {
        throw "Release mode requires trusted signing for the project executable and MSI"
    }
    Assert-File $SignToolPath "Microsoft signtool"
    Assert-File $SignCommand "Artifact signing command"
    if ($isDirty) {
        throw "Release mode requires a clean worktree for exact source correspondence"
    }
    if ([string]::IsNullOrWhiteSpace($SourceRevision)) {
        throw "Release mode requires CUAJONE_SOURCE_REVISION"
    }
    if ($SourceRevision -cne $gitHead) {
        throw "SourceRevision must exactly match repository HEAD $gitHead"
    }
    Assert-HttpsUrl $SourceArchiveUrl "Project source archive URL"
    Assert-Sha256 $SourceArchiveSha256 "Project source archive hash"
    Assert-HttpsUrl $FfmpegSourceArchiveUrl "FFmpeg corresponding-source archive URL"
    Assert-Sha256 $FfmpegSourceArchiveSha256 "FFmpeg corresponding-source archive hash"
    $productionParityReceipt = Assert-ProductionParityReceipt $ParityReceiptPath $gitHead "1.0.0"
} else {
    if (-not $isSignedBuild -and -not $AllowUnsignedPreview) {
        throw "Unsigned preview builds require the explicit -AllowUnsignedPreview switch"
    }
    if ([string]::IsNullOrWhiteSpace($SourceRevision)) {
        $SourceRevision = if ($isDirty) {
            "WORKTREE-PREVIEW-AT-$gitHead-DIRTY"
        } else {
            $gitHead
        }
    }
}

$fileVersionParts = @($FileVersion.Split('.') | ForEach-Object { [int]$_ })
if ($fileVersionParts[0] -gt 255 -or $fileVersionParts[1] -gt 255 -or $fileVersionParts[3] -gt 65535) {
    throw "FileVersion cannot be mapped to the Windows Installer major.minor.build ranges"
}
$msiVersion = "{0}.{1}.{2}" -f $fileVersionParts[0], $fileVersionParts[1], $fileVersionParts[3]

$dumpbin = Join-Path $ToolRoot "vs\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\dumpbin.exe"
$wix = Join-Path $WixToolRoot "wix.exe"
$openCvBin = Join-Path $ToolRoot "opencv\opencv\build\x64\vc16\bin"
$cudaBin = Join-Path $ToolRoot "cuda-runtime\nvidia\cuda_runtime\bin"
$cublasWheel = Join-Path $ToolRoot "downloads\nvidia_cublas_cu12-12.9.2.10-py3-none-win_amd64.whl"
$cudnnWheel = Join-Path $ToolRoot "downloads\nvidia_cudnn_cu12-9.24.0.43-py3-none-win_amd64.whl"
$cufftWheel = Join-Path $ToolRoot "downloads\nvidia_cufft_cu12-11.4.1.4-py3-none-win_amd64.whl"
$cublasRoot = Join-Path $ToolRoot "nvidia-libraries\cublas"
$cudnnRoot = Join-Path $ToolRoot "nvidia-libraries\cudnn"
$cufftRoot = Join-Path $ToolRoot "nvidia-libraries\cufft"
$cublasBin = Join-Path $cublasRoot "nvidia\cublas\bin"
$cudnnBin = Join-Path $cudnnRoot "nvidia\cudnn\bin"
$cufftBin = Join-Path $cufftRoot "nvidia\cufft\bin"
$tensorRtRoot = Join-Path $ToolRoot "tensorrt\TensorRT-11.1.0.106"
$tensorRtBin = Join-Path $tensorRtRoot "bin"
$onnxRuntimeRoot = Join-Path $ToolRoot "onnxruntime-win-x64-1.25.0"
$onnxRuntimeGpuDefaultRoot = Join-Path $ToolRoot "onnxruntime-win-x64-gpu-1.25.0"
$byteTrackCommit = "a865158906f6138465668810a98ffd918d95f9a3"
$eigenCommit = "3147391d946bb4b6c68edd901f2add6ac1f31f8c"
$byteTrackRoot = Join-Path $ToolRoot "dependencies\byte-track-eigen-$byteTrackCommit"
$eigenRoot = Join-Path $ToolRoot "dependencies\eigen-$eigenCommit"
if ([string]::IsNullOrWhiteSpace($OnnxRuntimeGpuRoot)) { $OnnxRuntimeGpuRoot = $onnxRuntimeGpuDefaultRoot }
$onnxRuntimeBin = Join-Path $onnxRuntimeRoot "lib"
$msvcCrt = Join-Path $ToolRoot "vs\VC\Redist\MSVC\14.44.35112\x64\Microsoft.VC143.CRT"
$searchDirectories = @($openCvBin, $cudaBin, $cublasBin, $cudnnBin, $cufftBin, $tensorRtBin, $onnxRuntimeBin, $msvcCrt)

Assert-File $dumpbin "MSVC dumpbin"
Assert-File $wix "WiX CLI"
Assert-File (Join-Path $byteTrackRoot "LICENSE") "ByteTrack-Eigen MIT license"
Assert-File (Join-Path $byteTrackRoot ".cuajone-source-receipt.json") "ByteTrack-Eigen source receipt"
Assert-File (Join-Path $eigenRoot "COPYING.MPL2") "Eigen MPL-2.0 license"
Assert-File (Join-Path $eigenRoot "COPYING.README") "Eigen licensing readme"
Assert-File (Join-Path $eigenRoot ".cuajone-source-receipt.json") "Eigen source receipt"
Ensure-OnnxRuntimeCudaPackage $OnnxRuntimeGpuRoot
Expand-VerifiedNvidiaWheel $cublasWheel "623f43027d40d44ceadf0043f002bd25cf353e8f13ce90b9a87057019f560661" $cublasRoot "nvidia\cublas\bin\cublasLt64_12.dll" "NVIDIA cuBLAS Windows wheel"
Expand-VerifiedNvidiaWheel $cudnnWheel "cbd41a0ab084422c936dc9fb2fc89be5ea9a85bc421c6f23d0243bdfc945fbef" $cudnnRoot "nvidia\cudnn\bin\cudnn64_9.dll" "NVIDIA cuDNN Windows wheel"
Expand-VerifiedNvidiaWheel $cufftWheel "8e5bfaac795e93f80611f807d42844e8e27e340e0cde270dcb6c65386d795b80" $cufftRoot "nvidia\cufft\bin\cufft64_11.dll" "NVIDIA cuFFT Windows wheel"
Assert-File (Join-Path $OnnxRuntimeGpuRoot "lib\onnxruntime_providers_cuda.dll") "ONNX Runtime CUDA provider"
foreach ($directory in $searchDirectories) {
    Assert-Directory $directory "Runtime dependency directory"
}
$cudaProvider = Join-Path $OnnxRuntimeGpuRoot "lib\onnxruntime_providers_cuda.dll"
$cudaProviderMissingDependencies = foreach ($dependency in Get-PeDependencies $cudaProvider $dumpbin) {
    if (Find-Dependency $dependency $searchDirectories) {
        continue
    }
    $systemPath = Join-Path ([Environment]::SystemDirectory) $dependency
    if ($dependency -match '^(api-ms-|ext-ms-)' -or (Test-Path -LiteralPath $systemPath -PathType Leaf)) {
        continue
    }
    $dependency
}
if ($cudaProviderMissingDependencies) {
    throw "ONNX Runtime CUDA provider has unresolved dependencies: $($cudaProviderMissingDependencies -join ', ')"
}

$tempRoot = Join-Path $ToolRoot "temp\wix"
$dotnetHome = Join-Path $ToolRoot "dotnet-home"
$nugetPackages = Join-Path $ToolRoot "cache\nuget"
foreach ($path in @($tempRoot, $dotnetHome, $nugetPackages)) {
    Ensure-Directory $path
}
$env:TEMP = $tempRoot
$env:TMP = $tempRoot
$env:DOTNET_CLI_HOME = $dotnetHome
$env:NUGET_PACKAGES = $nugetPackages
$env:DOTNET_CLI_TELEMETRY_OPTOUT = "1"
$env:DOTNET_SKIP_FIRST_TIME_EXPERIENCE = "1"

$wixReportedVersion = (& $wix --version).Trim()
if ($LASTEXITCODE -ne 0 -or $wixReportedVersion -notmatch "^$([regex]::Escape($WixVersion))(?:\+|$)") {
    throw "Expected WiX $WixVersion at $wix; got '$wixReportedVersion'"
}
Push-Location $WixToolRoot
try {
    $extensions = @(& $wix extension list)
} finally {
    Pop-Location
}
foreach ($extension in @("WixToolset.UI.wixext $WixVersion", "WixToolset.Util.wixext $WixVersion")) {
    if ($extensions -notcontains $extension) {
        throw "Required local WiX extension is missing: $extension"
    }
}

$nativeRoot = Join-Path $projectRoot "native"
$cmakeLists = Join-Path $nativeRoot "CMakeLists.txt"
$allCpp = @(Get-ChildItem -LiteralPath (Join-Path $nativeRoot "src") -Recurse -File -Filter "*.cpp").FullName
$allHeaders = @(Get-ChildItem -LiteralPath (Join-Path $nativeRoot "include") -Recurse -File -Include "*.hpp", "*.h").FullName
$launcherNames = @("launcher.cpp", "launcher_support.cpp", "launcher_support.hpp")
$probeNames = @("compute.cpp", "installer_custom_action.cpp", "compute.hpp")
$freshnessByBinary = @{
    $ReleaseExecutable = @($cmakeLists) + @(
        $allCpp | Where-Object {
            (Split-Path -Leaf $_) -notin @("launcher.cpp", "launcher_support.cpp", "installer_custom_action.cpp")
        }
    ) + @(
        $allHeaders | Where-Object {
            (Split-Path -Leaf $_) -ne "launcher_support.hpp"
        }
    )
    $LauncherExecutable = @($cmakeLists) + @(
        $allCpp + $allHeaders | Where-Object {
            (Split-Path -Leaf $_) -in $launcherNames
        }
    )
    $HardwareProbeCustomAction = @($cmakeLists) + @(
        $allCpp + $allHeaders | Where-Object {
            (Split-Path -Leaf $_) -in $probeNames
        }
    )
}
foreach ($ownedBinary in @($ReleaseExecutable, $LauncherExecutable, $HardwareProbeCustomAction)) {
    $sourceInputs = @($freshnessByBinary[$ownedBinary] | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | Select-Object -Unique)
    if ($sourceInputs.Count -eq 0) {
        throw "No native source inputs were resolved for $(Split-Path -Leaf $ownedBinary)"
    }
    $newerSource = $sourceInputs | Where-Object {
        (Test-Path -LiteralPath $_) -and
        (Get-Item -LiteralPath $_).LastWriteTimeUtc -gt (Get-Item -LiteralPath $ownedBinary).LastWriteTimeUtc
    }
    if ($newerSource) {
        throw "$(Split-Path -Leaf $ownedBinary) is older than native source input: $($newerSource -join ', ')"
    }
}

if ($isSignedBuild) {
    Assert-File $SignToolPath "Microsoft signtool"
    Assert-File $SignCommand "Artifact signing command"
    & $SignCommand -FilePath $ReleaseExecutable
    if (-not $?) {
        throw "Project executable signing command failed"
    }
    & $signatureVerifier -FilePath $ReleaseExecutable -SignToolPath $SignToolPath -VerifyOnly
    & $SignCommand -FilePath $LauncherExecutable
    if (-not $?) {
        throw "Launcher executable signing command failed"
    }
    & $signatureVerifier -FilePath $LauncherExecutable -SignToolPath $SignToolPath -VerifyOnly
    & $SignCommand -FilePath $HardwareProbeCustomAction
    if (-not $?) {
        throw "Hardware probe custom action signing command failed"
    }
    & $signatureVerifier -FilePath $HardwareProbeCustomAction -SignToolPath $SignToolPath -VerifyOnly
} elseif ((Get-AuthenticodeSignature -LiteralPath $ReleaseExecutable).Status -ne
    [System.Management.Automation.SignatureStatus]::NotSigned) {
    throw "Explicit unsigned preview expected the project executable to be NotSigned"
} elseif ((Get-AuthenticodeSignature -LiteralPath $LauncherExecutable).Status -ne
    [System.Management.Automation.SignatureStatus]::NotSigned) {
    throw "Explicit unsigned preview expected the launcher executable to be NotSigned"
} elseif ((Get-AuthenticodeSignature -LiteralPath $HardwareProbeCustomAction).Status -ne
    [System.Management.Automation.SignatureStatus]::NotSigned) {
    throw "Explicit unsigned preview expected the hardware probe custom action to be NotSigned"
}

$generatedParent = Split-Path -Parent $StageDir
Ensure-Directory $generatedParent
foreach ($path in @($WixBuildDir, $OutputDir, $VerificationRoot)) {
    $parent = Split-Path -Parent $path
    Ensure-Directory $parent
}
Ensure-Directory $SupersededOutputDir
$stageReceiptPath = Join-Path $generatedParent "stage-receipt.json"
$stageReceipt = $null
$stageReceiptLookup = $null
if ($FastPreview -and (Test-Path -LiteralPath $stageReceiptPath -PathType Leaf)) {
    try {
        $stageReceipt = Get-Content -LiteralPath $stageReceiptPath -Raw | ConvertFrom-Json
        if ($null -ne $stageReceipt.stagedFiles) {
            $stageReceiptLookup = @{}
            foreach ($stagedEntry in @($stageReceipt.stagedFiles)) {
                if (-not [string]::IsNullOrWhiteSpace($stagedEntry.stagedRelativePath)) {
                    $stageReceiptLookup[$stagedEntry.stagedRelativePath] = $stagedEntry
                }
            }
        }
    } catch {
        $stageReceipt = $null
        $stageReceiptLookup = $null
    }
}
if ($FastPreview) {
    if (-not (Test-Path -LiteralPath $StageDir -PathType Container)) {
        New-Item -ItemType Directory -Path $StageDir -Force | Out-Null
    }
} else {
    Reset-Directory $StageDir
}
Reset-Directory $WixBuildDir
Ensure-Directory $OutputDir
Ensure-Directory $VerificationRoot

$stageBin = New-Item -ItemType Directory -Path (Join-Path $StageDir "bin") -Force
$stageDocs = New-Item -ItemType Directory -Path (Join-Path $StageDir "docs") -Force
$stageLicenses = New-Item -ItemType Directory -Path (Join-Path $StageDir "licenses") -Force

$resolved = [ordered]@{}
$imports = [System.Collections.Generic.List[object]]::new()
$stagedSources = [System.Collections.Generic.List[object]]::new()
$queue = [System.Collections.Generic.Queue[string]]::new()
$sourceRoots = [ordered]@{
    repository = $projectRoot
    build = (Join-Path $ToolRoot "build")
    tool = $ToolRoot
}

$generatedStageRelativePaths = if ($StageOnly) {
    @(
        "NexoAIVision.ico"
        "build-metadata.json"
        "docs/SOURCE-OFFER.txt"
        "docs/MODEL-BUNDLE.txt"
        "licenses/Microsoft-VC-Runtime-REDISTRIBUTION-REFERENCE.txt"
        "licenses/NVIDIA-TensorRT-LICENSE-REFERENCE.txt"
        "SHA256SUMS.txt"
    )
} else {
    @(
        "NexoAIVision.ico"
        "build-metadata.json"
        "docs/SOURCE-OFFER.txt"
        "docs/MODEL-BUNDLE.txt"
        "docs/sbom.spdx.json"
        "licenses/Microsoft-VC-Runtime-REDISTRIBUTION-REFERENCE.txt"
        "licenses/NVIDIA-TensorRT-LICENSE-REFERENCE.txt"
        "SHA256SUMS.txt"
    )
}

function Copy-StagedInput(
    [string]$Source,
    [string]$Destination,
    [ValidateSet("repository", "build", "tool")]
    [string]$SourceScope,
    [string]$Reason
) {
    Assert-File $Source "Staging source"
    $resolvedSource = (Resolve-Path -LiteralPath $Source).Path
    $scopeRoot = [System.IO.Path]::GetFullPath($sourceRoots[$SourceScope]).TrimEnd('\')
    if (-not $resolvedSource.StartsWith("$scopeRoot\", [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Staging source is outside its declared $SourceScope scope: $resolvedSource"
    }

    $resolvedStage = [System.IO.Path]::GetFullPath($StageDir).TrimEnd('\')
    $resolvedDestination = [System.IO.Path]::GetFullPath($Destination)
    if (-not $resolvedDestination.StartsWith("$resolvedStage\", [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Staging destination is outside the stage: $resolvedDestination"
    }
    $sourceHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $resolvedSource).Hash.ToLowerInvariant()
    $stagedRelativePath = [System.IO.Path]::GetRelativePath($resolvedStage, $resolvedDestination).Replace('\', '/')
    $skipCopy = $false
    if ($null -ne $stageReceiptLookup) {
        $priorEntry = $stageReceiptLookup[$stagedRelativePath]
        if ($null -ne $priorEntry -and
            $priorEntry.stagedSha256 -ceq $sourceHash -and
            (Test-Path -LiteralPath $resolvedDestination -PathType Leaf)) {
            $existingHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $resolvedDestination).Hash.ToLowerInvariant()
            if ($existingHash -ceq $sourceHash) {
                $skipCopy = $true
            }
        }
    }
    if ($skipCopy) {
        $stagedHash = $sourceHash
    } else {
        $destinationParent = Split-Path -Parent $resolvedDestination
        if (-not (Test-Path -LiteralPath $destinationParent -PathType Container)) {
            New-Item -ItemType Directory -Path $destinationParent -Force | Out-Null
        }
        Copy-Item -LiteralPath $resolvedSource -Destination $resolvedDestination
        $stagedHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $resolvedDestination).Hash.ToLowerInvariant()
        if ($stagedHash -cne $sourceHash) {
            throw "Staging changed source bytes: $resolvedSource"
        }
    }

    $stagedSources.Add([ordered]@{
        sourceScope = $SourceScope
        sourcePath = $resolvedSource
        sourceRelativePath = [System.IO.Path]::GetRelativePath($scopeRoot, $resolvedSource).Replace('\', '/')
        stagedRelativePath = $stagedRelativePath
        sourceSha256 = $sourceHash
        stagedSha256 = $stagedHash
        bytePreserved = $true
        reason = $Reason
    })
}

function Copy-StagedTree(
    [string]$SourceRoot,
    [string]$DestinationRoot,
    [ValidateSet("repository", "build", "tool")]
    [string]$SourceScope,
    [string]$Reason
) {
    Assert-Directory $SourceRoot "Staging source directory"
    foreach ($file in Get-ChildItem -LiteralPath $SourceRoot -Recurse -File | Sort-Object FullName) {
        $relative = [System.IO.Path]::GetRelativePath($SourceRoot, $file.FullName)
        Copy-StagedInput $file.FullName (Join-Path $DestinationRoot $relative) $SourceScope $Reason
    }
}

function Add-StagedBinary(
    [string]$Source,
    [string]$Reason,
    [ValidateSet("build", "tool")]
    [string]$SourceScope
) {
    $name = Split-Path -Leaf $Source
    if ($resolved.Contains($name)) {
        return
    }
    Assert-X64Pe $Source $dumpbin
    $destination = Join-Path $stageBin.FullName $name
    Copy-StagedInput $Source $destination $SourceScope $Reason
    $sourceHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Source).Hash.ToLowerInvariant()
    $resolved[$name] = [pscustomobject]@{
        source = $Source
        reason = $Reason
        sha256 = $sourceHash
    }
    $queue.Enqueue($Source)
}

Add-StagedBinary $ReleaseExecutable "Application executable" "build"
Add-StagedBinary $LauncherExecutable "Graphical launcher executable" "build"
$ffmpegPlugin = Join-Path $openCvBin "opencv_videoio_ffmpeg4120_64.dll"
Assert-File $ffmpegPlugin "OpenCV FFmpeg videoio plugin"
$ffmpegSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $ffmpegPlugin).Hash.ToLowerInvariant()
$ffmpegMd5 = (Get-FileHash -Algorithm MD5 -LiteralPath $ffmpegPlugin).Hash.ToLowerInvariant()
if ($ffmpegSha256 -cne "a0f01e4ee5e97b4a513cd70f01fafadc0dd187ba5d1293cb7fc6b77e7d17c631" -or
    $ffmpegMd5 -cne "e5c6936240201064b15bcecf1816e8f4") {
    throw "OpenCV FFmpeg plugin does not match the documented OpenCV 4.12.0 binary"
}
Add-StagedBinary $ffmpegPlugin "OpenCV videoio plugin loaded by filename for RTSP and offline capture" "tool"

$systemDirectory = [Environment]::SystemDirectory
while ($queue.Count -gt 0) {
    $binary = $queue.Dequeue()
    foreach ($dependency in Get-PeDependencies $binary $dumpbin) {
        $resolvedPath = Find-Dependency $dependency $searchDirectories
        if ($resolvedPath) {
            Add-StagedBinary $resolvedPath "PE import required by $(Split-Path -Leaf $binary)" "tool"
            $imports.Add([pscustomobject]@{
                importer = Split-Path -Leaf $binary
                dependency = $dependency
                resolution = "app-local"
            })
            continue
        }

        $systemPath = Join-Path $systemDirectory $dependency
        if ($dependency -match '^(api-ms-|ext-ms-)' -or (Test-Path -LiteralPath $systemPath -PathType Leaf)) {
            $imports.Add([pscustomobject]@{
                importer = Split-Path -Leaf $binary
                dependency = $dependency
                resolution = "Windows system"
            })
            continue
        }
        throw "Unresolved PE dependency '$dependency' imported by '$binary'"
    }
}

$cudaPayloadRelativePaths = [System.Collections.Generic.List[string]]::new()
$onnxRuntimeGpuBin = Join-Path $OnnxRuntimeGpuRoot "lib"
foreach ($cudaLibrary in @(
        [pscustomobject]@{ directory = $onnxRuntimeGpuBin; source = "ONNX Runtime CUDA provider dependency" },
        [pscustomobject]@{ directory = $cublasBin; source = "NVIDIA cuBLAS runtime dependency" },
        [pscustomobject]@{ directory = $cudnnBin; source = "NVIDIA cuDNN runtime dependency" },
        [pscustomobject]@{ directory = $cufftBin; source = "NVIDIA cuFFT runtime dependency" }
    )) {
    foreach ($file in Get-ChildItem -LiteralPath $cudaLibrary.directory -File | Sort-Object Name) {
        if ($file.Extension -ine ".dll" -or $file.Name -in @(
                "onnxruntime.dll", "onnxruntime_providers_tensorrt.dll"
            ) -or $resolved.Contains($file.Name) -or $cudaPayloadRelativePaths.Contains("bin\" + $file.Name)) {
            continue
        }
        Copy-StagedInput $file.FullName (Join-Path $stageBin.FullName $file.Name) "tool" $cudaLibrary.source
        $cudaPayloadRelativePaths.Add(("bin\" + $file.Name))
    }
}
$hasCudaPayload = $cudaPayloadRelativePaths.Count -gt 0

$unexpectedTensorRt = @($resolved.Keys | Where-Object {
    $_ -match '^(nvonnxparser|nvinfer_plugin|nvinfer_builder|nvinfer_vc_plugin)'
})
if ($unexpectedTensorRt) {
    throw "Development/plugin TensorRT DLLs were staged without an approved runtime boundary: $($unexpectedTensorRt -join ', ')"
}

Copy-StagedInput (Join-Path $scriptRoot "README.md") (Join-Path $stageDocs.FullName "README.md") "repository" "Installer and IT runbook"
Copy-StagedInput (Join-Path $projectRoot "INSTALACION_WINDOWS.md") (Join-Path $stageDocs.FullName "INSTALACION_WINDOWS.md") "repository" "End-user Windows installation guide"
Copy-StagedInput (Join-Path $projectRoot "README.md") (Join-Path $stageDocs.FullName "PROJECT-README.md") "repository" "Project overview"
Copy-StagedInput (Join-Path $scriptRoot "THIRD_PARTY_NOTICES.md") (Join-Path $stageDocs.FullName "THIRD_PARTY_NOTICES.md") "repository" "Third-party notices"
Copy-StagedInput (Join-Path $scriptRoot "FFMPEG-SOURCE.md") (Join-Path $stageDocs.FullName "FFMPEG-SOURCE.md") "repository" "FFmpeg source provenance"
Copy-StagedInput (Join-Path $projectRoot "LICENSE") (Join-Path $stageLicenses.FullName "AGPL-3.0.txt") "repository" "Project license"
Copy-StagedInput (Join-Path $projectRoot "LICENSES.md") (Join-Path $stageDocs.FullName "LICENSES.md") "repository" "Project licensing boundaries"
Copy-StagedInput (Join-Path $projectRoot "SECURITY.md") (Join-Path $stageDocs.FullName "SECURITY.md") "repository" "Project security policy"
Copy-StagedTree (Join-Path $scriptRoot "licenses") (Join-Path $stageLicenses.FullName "FFmpeg-dependencies") "repository" "FFmpeg dependency license"
Copy-StagedInput (Join-Path $ToolRoot "opencv\opencv\LICENSE.txt") (Join-Path $stageLicenses.FullName "OpenCV-LICENSE.txt") "tool" "OpenCV license"
Copy-StagedInput (Join-Path $ToolRoot "opencv\opencv\LICENSE_FFMPEG.txt") (Join-Path $stageLicenses.FullName "OpenCV-FFmpeg-LGPL-2.1.txt") "tool" "OpenCV FFmpeg license"
$openCvNoticeDir = Join-Path $ToolRoot "opencv\opencv\build\etc\licenses"
Assert-Directory $openCvNoticeDir "OpenCV bundled third-party notices"
Copy-StagedTree $openCvNoticeDir (Join-Path $stageLicenses.FullName "OpenCV-third-party") "tool" "OpenCV bundled third-party notice"
$cudaLicense = Join-Path $ToolRoot "cuda-runtime\nvidia_cuda_runtime_cu12-12.9.79.dist-info\licenses\License.txt"
Assert-File $cudaLicense "CUDA runtime license"
Copy-StagedInput $cudaLicense (Join-Path $stageLicenses.FullName "NVIDIA-CUDA-License.txt") "tool" "CUDA runtime license"
$cublasLicense = Join-Path $cublasRoot "nvidia_cublas_cu12-12.9.2.10.dist-info\licenses\License.txt"
$cudnnLicense = Join-Path $cudnnRoot "nvidia_cudnn_cu12-9.24.0.43.dist-info\licenses\License.txt"
$cufftLicense = Join-Path $cufftRoot "nvidia_cufft_cu12-11.4.1.4.dist-info\licenses\License.txt"
Assert-File $cublasLicense "NVIDIA cuBLAS license"
Assert-File $cudnnLicense "NVIDIA cuDNN license"
Assert-File $cufftLicense "NVIDIA cuFFT license"
Copy-StagedInput $cublasLicense (Join-Path $stageLicenses.FullName "NVIDIA-cuBLAS-License.txt") "tool" "NVIDIA cuBLAS license"
Copy-StagedInput $cudnnLicense (Join-Path $stageLicenses.FullName "NVIDIA-cuDNN-License.txt") "tool" "NVIDIA cuDNN license"
Copy-StagedInput $cufftLicense (Join-Path $stageLicenses.FullName "NVIDIA-cuFFT-License.txt") "tool" "NVIDIA cuFFT license"
Copy-StagedInput (Join-Path $tensorRtRoot "doc\README.txt") (Join-Path $stageLicenses.FullName "NVIDIA-TensorRT-README.txt") "tool" "TensorRT redistribution and license reference"
Copy-StagedInput (Join-Path $onnxRuntimeRoot "LICENSE") (Join-Path $stageLicenses.FullName "ONNX-Runtime-LICENSE.txt") "tool" "ONNX Runtime MIT license"
Copy-StagedInput (Join-Path $onnxRuntimeRoot "ThirdPartyNotices.txt") (Join-Path $stageLicenses.FullName "ONNX-Runtime-ThirdPartyNotices.txt") "tool" "ONNX Runtime third-party notices"
Copy-StagedInput (Join-Path $byteTrackRoot "LICENSE") (Join-Path $stageLicenses.FullName "ByteTrack-Eigen-MIT.txt") "tool" "Statically linked ByteTrack-Eigen license"
Copy-StagedInput (Join-Path $eigenRoot "COPYING.MPL2") (Join-Path $stageLicenses.FullName "Eigen-MPL-2.0.txt") "tool" "Statically linked Eigen license"
Copy-StagedInput (Join-Path $eigenRoot "COPYING.README") (Join-Path $stageLicenses.FullName "Eigen-COPYING-README.txt") "tool" "Eigen licensing scope and EIGEN_MPL2_ONLY notice"

$projectSourceAvailability = if ($BuildMode -eq "Release") {
    "$SourceArchiveUrl`nSHA-256: $($SourceArchiveSha256.ToLowerInvariant())"
} else {
    "NOT PUBLISHED FOR THIS INTERNAL PREVIEW. Exact external distribution is blocked."
}
$ffmpegSourceAvailability = if ($BuildMode -eq "Release") {
    "$FfmpegSourceArchiveUrl`nSHA-256: $($FfmpegSourceArchiveSha256.ToLowerInvariant())"
} else {
    "NOT PUBLISHED FOR THIS INTERNAL PREVIEW. External distribution is blocked."
}
$includeModelBundle = -not $SkipModelBundle
$hasModels = $false
$ppeEngineSha256 = $null
$poseEngineSha256 = $null

if ($PpeEnginePath -or $PoseEnginePath) {
    if (-not $PpeEnginePath) { throw "-PpeEnginePath is required when providing pre-built engines" }
    if (-not $PoseEnginePath) { throw "-PoseEnginePath is required when providing pre-built engines" }
    Assert-File $PpeEnginePath "Pre-built PPE TensorRT engine"
    Assert-File $PoseEnginePath "Pre-built pose TensorRT engine"
    $stageModelsDir = New-Item -ItemType Directory -Path (Join-Path $StageDir "bin\models") -Force
    Copy-StagedInput $PpeEnginePath (Join-Path $stageModelsDir.FullName "ppe.engine") "tool" "Pre-built PPE TensorRT engine"
    Copy-StagedInput $PoseEnginePath (Join-Path $stageModelsDir.FullName "pose.engine") "tool" "Pre-built pose TensorRT engine"
    $ppeEngineSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $PpeEnginePath).Hash.ToLowerInvariant()
    $poseEngineSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $PoseEnginePath).Hash.ToLowerInvariant()
    $hasModels = $true
    $modelBundleSource = "pre-built"
    @"
NexoAI Vision model bundle

Build mode: $BuildMode
Bundle source: Pre-built external engines
Bundle included: yes

PPE engine: $PpeEnginePath
PPE engine SHA-256: $ppeEngineSha256

Pose engine: $PoseEnginePath
Pose engine SHA-256: $poseEngineSha256
"@ | Set-Content -LiteralPath (Join-Path $stageDocs.FullName "MODEL-BUNDLE.txt") -Encoding UTF8
} elseif ($includeModelBundle) {
    $pythonExecutable = Resolve-FirstExistingPath @(
        $PythonExecutable,
        ".venv\Scripts\python.exe"
    ) "Python executable for ONNX export"
    $ppeModelSource = Resolve-ModelSource $PpeModelPath @(
        "best_ppe.pt"
    ) "Bundled PPE source model"
    $poseModelSource = Resolve-ModelSource $PoseModelPath @(
        "yolo26s-pose.pt",
        "yolo26n-pose.pt"
    ) "Bundled pose source model"
    $exportRoot = Join-Path $ToolRoot "installer\model-export\cache"
    $ppeExport = Get-OrExportOnnxModel $pythonExecutable $ppeModelSource $exportRoot "ppe" "detect" -Refresh:$RefreshModelExport
    $ppeOnnxResult = $ppeExport.exported
    $ppeManifest = $ppeExport.manifest
    $poseExport = Get-OrExportOnnxModel $pythonExecutable $poseModelSource $exportRoot "pose" "pose" -Refresh:$RefreshModelExport
    $poseOnnxResult = $poseExport.exported
    $poseManifest = $poseExport.manifest
    Write-Host "PPE ONNX: $(if ($ppeExport.cacheHit) { 'verified cache hit' } else { 'exported' })"
    Write-Host "Pose ONNX: $(if ($poseExport.cacheHit) { 'verified cache hit' } else { 'exported' })"

    $null = New-Item -ItemType Directory -Path (Join-Path $StageDir "bin\models") -Force
    Copy-StagedInput $ppeOnnxResult.onnx (Join-Path $StageDir "bin\models\ppe.onnx") "tool" "Auto-exported PPE ONNX model"
    Copy-StagedInput $ppeManifest.path (Join-Path $StageDir "bin\models\ppe.onnx.manifest.json") "tool" "PPE ONNX model manifest"
    Copy-StagedInput $poseOnnxResult.onnx (Join-Path $StageDir "bin\models\pose.onnx") "tool" "Auto-exported pose ONNX model"
    Copy-StagedInput $poseManifest.path (Join-Path $StageDir "bin\models\pose.onnx.manifest.json") "tool" "Pose ONNX model manifest"
    $hasModels = $true
    $modelBundleSource = "content-addressed-onnx"

    @"
NexoAI Vision model bundle

Build mode: $BuildMode
Bundle source: Content-addressed ONNX export cache (re-exported only when source or recipe changes)
Bundle included: yes

PPE source model: $($ppeOnnxResult.source)
PPE source SHA-256: $($ppeOnnxResult.sourceSha256)
PPE ONNX: $($ppeOnnxResult.onnx)
PPE ONNX SHA-256: $($ppeOnnxResult.onnxSha256)
PPE ONNX manifest SHA-256: $($ppeManifest.sha256)

Pose source model: $($poseOnnxResult.source)
Pose source SHA-256: $($poseOnnxResult.sourceSha256)
Pose ONNX: $($poseOnnxResult.onnx)
Pose ONNX SHA-256: $($poseOnnxResult.onnxSha256)
Pose ONNX manifest SHA-256: $($poseManifest.sha256)
"@ | Set-Content -LiteralPath (Join-Path $stageDocs.FullName "MODEL-BUNDLE.txt") -Encoding UTF8
} else {
    @"
NexoAI Vision model bundle

Build mode: $BuildMode
Bundle included: no

The MSI was built without AI models. The launcher will use bundled ONNX or TensorRT
models when present under INSTALLFOLDER\bin\models; otherwise browse to externally
supplied model files before starting the runtime.
"@ | Set-Content -LiteralPath (Join-Path $stageDocs.FullName "MODEL-BUNDLE.txt") -Encoding UTF8
}

if ($FastPreview) {
    $currentStagedPaths = [System.Collections.Generic.HashSet[string]]::new(
        [System.StringComparer]::OrdinalIgnoreCase)
    foreach ($stagedEntry in $stagedSources) {
        $null = $currentStagedPaths.Add([string]$stagedEntry.stagedRelativePath)
    }
    foreach ($stagedFile in Get-ChildItem -LiteralPath $StageDir -Recurse -File) {
        $relativePath = [System.IO.Path]::GetRelativePath($StageDir, $stagedFile.FullName).Replace('\', '/')
        if (-not $currentStagedPaths.Contains($relativePath) -and $relativePath -notin $generatedStageRelativePaths) {
            Remove-Item -LiteralPath $stagedFile.FullName -Force
        }
    }
}

@"
NexoAI Vision source offer and release correspondence

Application version: $Version
Numeric file version: $FileVersion
MSI product version: $msiVersion
Build mode: $BuildMode
Source repository: $sourceRepository
Source revision/correspondence: $SourceRevision

Project corresponding source archive:
$projectSourceAvailability

FFmpeg/OpenCV wrapper corresponding-source archive:
$ffmpegSourceAvailability

The original project source is AGPL-3.0-only. The complete license is installed
as licenses/AGPL-3.0.txt. Third-party components, model weights, and datasets keep
their upstream terms and are not relicensed by the project.

For an external object-code release, source must be offered with equivalent access
and retained for the applicable period. See docs/FFMPEG-SOURCE.md and
docs/THIRD_PARTY_NOTICES.md.
"@ | Set-Content -LiteralPath (Join-Path $stageDocs.FullName "SOURCE-OFFER.txt") -Encoding UTF8

@"
TensorRT 11.1 runtime license reference

The TensorRT archive points to the current NVIDIA TensorRT Software License Agreement:
https://docs.nvidia.com/deeplearning/tensorrt/latest/reference/sla.html

The TensorRT supplement identifies runtime .dll files as distributable subject to the complete agreement and its distribution requirements. This reference is not a replacement for the agreement and does not grant additional rights.
"@ | Set-Content -LiteralPath (Join-Path $stageLicenses.FullName "NVIDIA-TensorRT-LICENSE-REFERENCE.txt") -Encoding UTF8

@"
Microsoft Visual C++ Runtime redistribution reference

The packaged MSVC runtime DLLs are unmodified files from:
$msvcCrt

Official Visual Studio 2022 redistribution terms and the distributable-file list:
https://learn.microsoft.com/visualstudio/releases/2022/redistribution

This reference is not a replacement for the applicable Visual Studio license terms and does not grant additional rights.
"@ | Set-Content -LiteralPath (Join-Path $stageLicenses.FullName "Microsoft-VC-Runtime-REDISTRIBUTION-REFERENCE.txt") -Encoding UTF8

if (-not $StageOnly) {
$sbomPackages = @($resolved.GetEnumerator() | Sort-Object Key | ForEach-Object {
    [ordered]@{
        SPDXID = "SPDXRef-Binary-$((Get-StableHex $_.Key).Substring(0, 16))"
        name = $_.Key
        versionInfo = if ($_.Key -ceq "onnxruntime.dll") { $onnxRuntimeVersion } else { "NOASSERTION" }
        downloadLocation = if ($_.Key -ceq "onnxruntime.dll") { $onnxRuntimeAssetUrl } else { "NOASSERTION" }
        filesAnalyzed = $false
        checksums = @([ordered]@{ algorithm = "SHA256"; checksumValue = $_.Value.sha256 })
        licenseConcluded = "NOASSERTION"
        licenseDeclared = if ($_.Key -ceq "onnxruntime.dll") { "MIT" } else { "NOASSERTION" }
        copyrightText = "NOASSERTION"
    }
})
$sbomPackages += @(
    [ordered]@{
        SPDXID = "SPDXRef-Static-ByteTrack-Eigen"
        name = "byte-track-eigen"
        versionInfo = "2.1.0"
        downloadLocation = "https://codeload.github.com/cj-mills/byte-track-eigen/zip/$byteTrackCommit"
        filesAnalyzed = $false
        checksums = @([ordered]@{ algorithm = "SHA256"; checksumValue = "e5a075df5e8b4ed4bb7436ffe7fe0f4cee5c6a6663112d6a1c47a99ffb704d88" })
        licenseConcluded = "MIT"
        licenseDeclared = "MIT"
        copyrightText = "Copyright (c) 2023 Christian J. Mills"
    },
    [ordered]@{
        SPDXID = "SPDXRef-Static-Eigen"
        name = "eigen"
        versionInfo = "3.4.0"
        downloadLocation = "https://gitlab.com/libeigen/eigen/-/archive/$eigenCommit/eigen-$eigenCommit.zip"
        filesAnalyzed = $false
        checksums = @([ordered]@{ algorithm = "SHA256"; checksumValue = "9eec4ec4e5e459b2f59dbbaa4280e1bb3ee61cccd8a7c0af0321d29d95fece9e" })
        licenseConcluded = "MPL-2.0"
        licenseDeclared = "MPL-2.0"
        copyrightText = "NOASSERTION"
    }
)
$releasePackageName = Split-Path -Leaf $ReleaseExecutable
$releasePackage = @($sbomPackages | Where-Object { $_.name -ceq $releasePackageName })
if ($releasePackage.Count -ne 1) {
    throw "Could not identify the application package for static-dependency SBOM relationships"
}
$sbom = [ordered]@{
    spdxVersion = "SPDX-2.3"
    dataLicense = "CC0-1.0"
    SPDXID = "SPDXRef-DOCUMENT"
    name = "NexoAI-Vision-$Version-x64"
    documentNamespace = "https://github.com/jeanpaulandradeleiva-crypto/CAMARAS-IP-CUAJONE/sbom/$Version/$([Guid]::NewGuid())"
    creationInfo = [ordered]@{
        created = [DateTime]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ")
        creators = @("Tool: installer/native/build-installer.ps1")
    }
    packages = $sbomPackages
    relationships = @(
        [ordered]@{ spdxElementId = $releasePackage[0].SPDXID; relationshipType = "STATIC_LINK"; relatedSpdxElement = "SPDXRef-Static-ByteTrack-Eigen" },
        [ordered]@{ spdxElementId = $releasePackage[0].SPDXID; relationshipType = "STATIC_LINK"; relatedSpdxElement = "SPDXRef-Static-Eigen" }
    )
}
$sbom | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $stageDocs.FullName "sbom.spdx.json") -Encoding UTF8
}

$iconPath = Join-Path $StageDir "NexoAIVision.ico"
& $iconGenerator -OutputPath $iconPath
if ($LASTEXITCODE -ne 0) {
    throw "Icon generation failed"
}
Assert-File $iconPath "Generated application icon"

$metadata = [ordered]@{
    product = "NexoAI Vision"
    appVersion = $Version
    fileVersion = $FileVersion
    msiVersion = $msiVersion
    architecture = "x64"
    configuration = "Release"
    buildUtc = [DateTime]::UtcNow.ToString("o")
    releaseExecutable = $ReleaseExecutable
    releaseExecutableSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $ReleaseExecutable).Hash.ToLowerInvariant()
    launcherExecutable = $LauncherExecutable
    launcherExecutableSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $LauncherExecutable).Hash.ToLowerInvariant()
    hardwareProbeCustomAction = $HardwareProbeCustomAction
    hardwareProbeCustomActionSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $HardwareProbeCustomAction).Hash.ToLowerInvariant()
    onnxRuntime = [ordered]@{
        version = $onnxRuntimeVersion
        root = $onnxRuntimeRoot
        assetUrl = $onnxRuntimeAssetUrl
        assetSha256 = $onnxRuntimeAssetSha256
        executionProvider = "CPUExecutionProvider"
    }
    runtimeSecurityPolicy = [ordered]@{
        hardwareProbeSchemaVersion = 2
        minimumCudaDriverApiVersion = 12090
        minimumTensorRtComputeCapability = "7.5"
        onnxManifestSchemaVersion = 1
        onnxExternalDataAllowed = $false
        onnxCustomOperatorsAllowed = $false
        maximumOnnxModelBytes = 268435456
        maximumTensorRtEngineBytes = 1073741824
        maximumImageDimension = 4096
        maximumOutputElements = 16777216
        maximumTensorBytes = 268435456
    }
    toolRoot = $ToolRoot
    dumpbin = $dumpbin
    wixToolset = $wixReportedVersion
    upgradeCode = $upgradeCode
    installFolderDefault = "C:\Program Files\NexoAI Vision"
    dataFolder = "C:\ProgramData\NexoAI Vision\runtime"
    licenseStatus = "Original project source is AGPL-3.0-only; third-party artifacts, models, and datasets retain separate upstream terms"
    releaseStatus = if ($BuildMode -eq "Release") {
        "Open-source release with exact source archives and trusted Authenticode required"
    } else {
        "Internal preview; public trust is not implied; real-engine operation not validated"
    }
    buildMode = $BuildMode
    fastPreview = [bool]$FastPreview
    stageOnly = [bool]$StageOnly
    cabinetCompressionLevel = $cabinetCompressionLevel
    sourceRepository = $sourceRepository
    sourceRevision = $SourceRevision
    sourceArchiveUrl = $SourceArchiveUrl
    sourceArchiveSha256 = $SourceArchiveSha256
    ffmpegSourceArchiveUrl = $FfmpegSourceArchiveUrl
    ffmpegSourceArchiveSha256 = $FfmpegSourceArchiveSha256
    ffmpegBinarySha256 = $ffmpegSha256
    ffmpegBinaryMd5 = $ffmpegMd5
    ffmpegUpstreamVersion = "n4.4.6"
    ffmpegOpenCvBinariesCommit = "ea9240e39bc0d6a69d2b1f0ba4513bdc7612a41e"
    modelBundle = if ($PpeEnginePath -or $PoseEnginePath) {
        [ordered]@{
            included = $true
            source = "pre-built"
            installRoot = "INSTALLFOLDER\bin\models"
            ppe = [ordered]@{
                source = $PpeEnginePath
                sourceSha256 = $ppeEngineSha256
                artifact = "bin/models/ppe.engine"
            }
            pose = [ordered]@{
                source = $PoseEnginePath
                sourceSha256 = $poseEngineSha256
                artifact = "bin/models/pose.engine"
            }
        }
    } elseif ($includeModelBundle) {
        [ordered]@{
            included = $true
            source = "auto-exported-onnx"
            installRoot = "INSTALLFOLDER\bin\models"
            ppe = [ordered]@{
                source = $ppeOnnxResult.source
                sourceSha256 = $ppeOnnxResult.sourceSha256
                artifact = "bin/models/ppe.onnx"
                artifactSha256 = $ppeOnnxResult.onnxSha256
                manifest = "bin/models/ppe.onnx.manifest.json"
                manifestSha256 = $ppeManifest.sha256
            }
            pose = [ordered]@{
                source = $poseOnnxResult.source
                sourceSha256 = $poseOnnxResult.sourceSha256
                artifact = "bin/models/pose.onnx"
                artifactSha256 = $poseOnnxResult.onnxSha256
                manifest = "bin/models/pose.onnx.manifest.json"
                manifestSha256 = $poseManifest.sha256
            }
        }
    } else {
        [ordered]@{
            included = $false
            reason = if ($SkipModelBundle) {
                "Model bundle was explicitly skipped"
            } else {
                "No model bundle was requested"
            }
        }
    }
    projectLicenseSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $projectRoot "LICENSE")).Hash.ToLowerInvariant()
    signingPolicy = if ($isInternalPilotSigning) {
        "Private Authenticode for enrolled internal pilot machines"
    } elseif ($isSignedBuild) {
        "Publicly trusted Authenticode required and verified"
    } else {
        "Explicit unsigned internal preview"
    }
    thirdPartyBinariesResigned = $false
    acceptanceScope = Get-AcceptanceScope -FastPreview:$FastPreview -StageOnly:$StageOnly
    parityGate = if ($BuildMode -eq "Release") {
        [ordered]@{
            receiptVersion = $productionParityReceipt.receipt_version
            contractVersion = $productionParityReceipt.contract_version
            sourceCommit = $productionParityReceipt.source_commit
            scope = $productionParityReceipt.scope
            receiptSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $ParityReceiptPath).Hash.ToLowerInvariant()
        }
    } else {
        [ordered]@{
            required = $false
            reason = "Preview does not claim authorized engine/model parity"
        }
    }
    stagingProvenanceVersion = 1
    sourceRoots = $sourceRoots
    sourceProvenance = @($stagedSources)
    generatedStagePaths = $generatedStageRelativePaths
    stagedBinaries = @($resolved.GetEnumerator() | ForEach-Object {
        [ordered]@{
            name = $_.Key
            source = $_.Value.source
            sha256 = $_.Value.sha256
            reason = $_.Value.reason
        }
    })
    imports = @($imports)
}
$metadata | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $StageDir "build-metadata.json") -Encoding UTF8

Assert-NoForbiddenPayloadFiles $StageDir "installer stage"
$manifestPath = Join-Path $StageDir "SHA256SUMS.txt"
$manifestLines = foreach ($file in Get-ChildItem -LiteralPath $StageDir -Recurse -File | Sort-Object FullName) {
    if ($file.FullName -eq $manifestPath) {
        continue
    }
    $relative = [System.IO.Path]::GetRelativePath($StageDir, $file.FullName).Replace('\', '/')
    "{0}  {1}" -f (Get-FileHash -Algorithm SHA256 -LiteralPath $file.FullName).Hash.ToLowerInvariant(), $relative
}
$manifestLines | Set-Content -LiteralPath $manifestPath -Encoding ASCII
Assert-NoForbiddenPayloadFiles $StageDir "installer stage"

$stageReceiptDocument = [ordered]@{
    receiptVersion = 1
    generatedUtc = [DateTime]::UtcNow.ToString("o")
    version = $Version
    fileVersion = $FileVersion
    buildMode = $BuildMode
    fastPreview = [bool]$FastPreview
    stageOnly = [bool]$StageOnly
    stagedFiles = @($stagedSources | ForEach-Object {
        [ordered]@{
            sourceScope = $_.sourceScope
            sourcePath = $_.sourcePath
            sourceRelativePath = $_.sourceRelativePath
            stagedRelativePath = $_.stagedRelativePath
            stagedSha256 = $_.stagedSha256
        }
    })
}
$stageReceiptDocument | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $stageReceiptPath -Encoding UTF8

if ($StageOnly) {
    Write-Host "Portable layout ready: $(Join-Path $StageDir 'bin\cuajone_launcher.exe')"
    return [pscustomobject]@{
        StageOnly = $true
        FastPreview = $FastPreview
        Stage = $StageDir
        Manifest = $manifestPath
        Receipt = $stageReceiptPath
        StagedBinaries = @($resolved.Keys)
        StagedSourceFiles = @($stagedSources).Count
    }
}

$licenseText = Get-Content -LiteralPath (Join-Path $projectRoot "LICENSE") -Raw
$rtfText = $licenseText.Replace('\', '\\').Replace('{', '\{').Replace('}', '\}')
$rtfText = $rtfText -replace "\r?\n", "\par`r`n"
$licenseRtf = Join-Path $WixBuildDir "AGPL-3.0.rtf"
"{\rtf1\ansi\deff0{\fonttbl{\f0 Courier New;}}\fs18 $rtfText}" |
    Set-Content -LiteralPath $licenseRtf -Encoding ASCII

$payloadSource = Join-Path $WixBuildDir "Payload.wxs"
Write-PayloadSource $StageDir $payloadSource -ComponentGroupName "PayloadComponents" -ExcludePrefix "bin\models\" -ExcludeRelativePaths $cudaPayloadRelativePaths
Assert-File $payloadSource "Generated WiX payload source"
if ($hasModels) {
    $modelsPayloadSource = Join-Path $WixBuildDir "Models.wxs"
    Write-PayloadSource $StageDir $modelsPayloadSource -ComponentGroupName "ModelComponents" -IncludePrefix "bin\models\"
    Assert-File $modelsPayloadSource "Generated WiX model source"
}
$cudaPayloadSource = $null
if ($hasCudaPayload) {
    $cudaPayloadSource = Join-Path $WixBuildDir "CudaPayload.wxs"
    Write-PayloadSource $StageDir $cudaPayloadSource -ComponentGroupName "CudaPayloadComponents" -IncludeRelativePaths $cudaPayloadRelativePaths -ComponentCondition 'CUDA_READY="1"'
}

$outputBaseFilename = if ($BuildMode -eq "Preview") {
    "NexoAIVision-$Version-x64-Internal"
} else {
    "NexoAIVision-$Version-x64"
}
$installerPath = Join-Path $OutputDir "$outputBaseFilename.msi"
$wixPdb = Join-Path $WixBuildDir "$outputBaseFilename.wixpdb"
$wixIntermediate = Join-Path $WixBuildDir "obj"
Ensure-Directory $wixIntermediate
$existingCandidates = @(Get-ChildItem -LiteralPath $OutputDir -File | Where-Object {
    $_.Name -like 'NexoAIVision-*-x64*.msi' -or
    $_.Name -like 'NexoAIVision-*-x64*.msi.sha256' -or
    $_.Name -like 'CuajonePPEMonitor-*-x64*.msi' -or
    $_.Name -like 'CuajonePPEMonitor-*-x64*.msi.sha256'
})
if ($existingCandidates.Count -gt 0) {
    $archiveName = "{0}-{1}" -f [DateTime]::UtcNow.ToString("yyyyMMdd-HHmmssfff"), $Version
    $archiveDirectory = Join-Path $SupersededOutputDir $archiveName
    New-Item -ItemType Directory -Path $archiveDirectory | Out-Null
    foreach ($candidate in $existingCandidates) {
        Move-Item -LiteralPath $candidate.FullName -Destination $archiveDirectory
    }
}

$arpComments = if ($BuildMode -eq "Preview") {
    "Open-source AGPL-3.0 internal preview. Pilot signing and real-engine validation limitations apply."
} else {
    "Open-source AGPL-3.0 release. Third-party components retain their own license terms."
}
$wixSourceFiles = @($packageSource, $payloadSource)
if ($hasModels) { $wixSourceFiles += $modelsPayloadSource }
if ($hasCudaPayload) { $wixSourceFiles += $cudaPayloadSource }
$wixArguments = @("build") + $wixSourceFiles + @(
    "-arch", "x64",
    "-d", "StageDir=$StageDir",
    "-d", "HasModels=$(if ($hasModels) { '1' } else { '0' })",
    "-d", "HasCudaPayload=$(if ($hasCudaPayload) { '1' } else { '0' })",
    "-d", "AppVersion=$Version",
    "-d", "MsiVersion=$msiVersion",
    "-d", "ArpComments=$arpComments",
    "-d", "LicenseRtf=$licenseRtf",
    "-d", "CabinetCompressionLevel=$cabinetCompressionLevel",
    "-d", "HardwareProbeCA=$HardwareProbeCustomAction",
    "-ext", "WixToolset.UI.wixext",
    "-ext", "WixToolset.Util.wixext",
    "-intermediatefolder", $wixIntermediate,
    "-pdb", $wixPdb,
    "-out", $installerPath
)
Push-Location $WixToolRoot
try {
    & $wix @wixArguments
    if ($LASTEXITCODE -ne 0) {
        throw "WiX compilation failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}
Assert-File $installerPath "Compiled MSI"

if ($isSignedBuild) {
    & $SignCommand -FilePath $installerPath
    if (-not $?) {
        throw "MSI signing command failed"
    }
    & $signatureVerifier -FilePath $installerPath -SignToolPath $SignToolPath -VerifyOnly
} elseif ((Get-AuthenticodeSignature -LiteralPath $installerPath).Status -ne
    [System.Management.Automation.SignatureStatus]::NotSigned) {
    throw "Explicit unsigned preview expected the MSI to be NotSigned"
}

$installerHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $installerPath).Hash.ToLowerInvariant()
$sidecarPath = "$installerPath.sha256"
"$installerHash  $(Split-Path -Leaf $installerPath)" | Set-Content -LiteralPath $sidecarPath -Encoding ASCII
$activeCandidates = @(Get-ChildItem -LiteralPath $OutputDir -File | Where-Object {
    $_.Name -like 'NexoAIVision-*-x64*.msi' -or
    $_.Name -like 'NexoAIVision-*-x64*.msi.sha256' -or
    $_.Name -like 'CuajonePPEMonitor-*-x64*.msi' -or
    $_.Name -like 'CuajonePPEMonitor-*-x64*.msi.sha256'
})
$expectedActiveNames = @(
    (Split-Path -Leaf $installerPath)
    (Split-Path -Leaf $sidecarPath)
)
if ($activeCandidates.Count -ne 2 -or @($activeCandidates.Name | Where-Object {
        $_ -notin $expectedActiveNames
    }).Count -ne 0) {
    throw "Active output must contain exactly the current MSI and SHA-256 sidecar"
}

$verificationParameters = @{
    InstallerPath = $installerPath
    StageDir = $StageDir
    AcceptanceRoot = $VerificationRoot
    WixToolRoot = $WixToolRoot
    SignToolPath = $SignToolPath
    ExpectedSignatureStatus = if ($isSignedBuild) { "Signed" } else { "NotSigned" }
}
if ($FastPreview) {
    $verificationParameters.FastPreview = $true
}
if ($isInternalPilotSigning) {
    $verificationParameters.AllowInternalPilotTrust = $true
    $verificationParameters.CertificateThumbprint = $env:CUAJONE_CERTIFICATE_SHA1
    $verificationParameters.PilotRootCertificatePath = $env:CUAJONE_PILOT_ROOT_CER
}
$verification = & $packageVerifier @verificationParameters

$signature = Get-AuthenticodeSignature -LiteralPath $installerPath
[pscustomobject]@{
    Installer = $installerPath
    SHA256 = $installerHash
    SizeBytes = (Get-Item -LiteralPath $installerPath).Length
    SignatureStatus = if ($isInternalPilotSigning) {
        "ValidForInternalPilot"
    } else {
        $signature.Status.ToString()
    }
    AppVersion = $Version
    MsiVersion = $msiVersion
    UpgradeCode = $upgradeCode
    Stage = $StageDir
    Manifest = $manifestPath
    Sidecar = $sidecarPath
    WixPdb = $wixPdb
    FastPreview = $FastPreview
    StageOnly = $StageOnly
    CabinetCompressionLevel = $cabinetCompressionLevel
    StagedBinaries = @($resolved.Keys)
    Verification = $verification
}
