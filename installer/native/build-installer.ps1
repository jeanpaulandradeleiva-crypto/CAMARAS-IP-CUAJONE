# SPDX-License-Identifier: AGPL-3.0-only

[CmdletBinding()]
param(
    [ValidatePattern('^\d+\.\d+\.\d+(-[0-9A-Za-z.-]+)?$')]
    [string]$Version = "0.1.0-internal.3",

    [ValidatePattern('^\d+\.\d+\.\d+\.\d+$')]
    [string]$FileVersion = "0.1.0.3",

    [ValidateSet("Preview", "Release")]
    [string]$BuildMode = "Release",

    [switch]$AllowUnsignedPreview,

    [string]$ToolRoot = "D:\DevTools\CuajoneNative",
    [string]$ReleaseExecutable = "D:\DevTools\CuajoneNative\build\windows-msvc\Release\cuajone_native.exe",
    [string]$StageDir = "D:\DevTools\CuajoneNative\installer\stage",
    [string]$OutputDir = "D:\DevTools\CuajoneNative\installer\output",
    [string]$SourceRevision = $env:CUAJONE_SOURCE_REVISION,
    [string]$SourceArchiveUrl = $env:CUAJONE_SOURCE_ARCHIVE_URL,
    [string]$SourceArchiveSha256 = $env:CUAJONE_SOURCE_ARCHIVE_SHA256,
    [string]$FfmpegSourceArchiveUrl = $env:CUAJONE_FFMPEG_SOURCE_ARCHIVE_URL,
    [string]$FfmpegSourceArchiveSha256 = $env:CUAJONE_FFMPEG_SOURCE_ARCHIVE_SHA256,
    [string]$SignToolPath = $env:CUAJONE_SIGNTOOL_PATH,
    [string]$PeSignCommand = $env:CUAJONE_PE_SIGN_COMMAND,
    [string]$SignToolName = $env:CUAJONE_SIGNTOOL_NAME,
    [string]$SignToolCommand = $env:CUAJONE_SIGNTOOL_COMMAND
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptRoot = $PSScriptRoot
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $scriptRoot "..\..")).Path
$installerScript = Join-Path $scriptRoot "CuajonePpeMonitor.iss"
$iconGenerator = Join-Path $scriptRoot "generate-icon.ps1"
$signatureVerifier = Join-Path $scriptRoot "sign-release.ps1"
$sourceRepository = "https://github.com/jeanpaulandradeleiva-crypto/CAMARAS-IP-CUAJONE"

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

function Reset-Directory([string]$Path) {
    Assert-DDrivePath $Path "Generated directory"
    $parent = Split-Path -Parent $Path
    Assert-Directory $parent "Generated directory parent"
    if (Test-Path -LiteralPath $Path) {
        Remove-Item -LiteralPath $Path -Recurse -Force
    }
    New-Item -ItemType Directory -Path $Path | Out-Null
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
        throw "Forbidden files entered the installer stage: $($violations -join ', ')"
    }
}

Assert-Directory $ToolRoot "Native tool root"
Assert-DDrivePath $StageDir "Staging directory"
Assert-DDrivePath $OutputDir "Installer output directory"
Assert-File $ReleaseExecutable "Release executable"
Assert-File $installerScript "Inno Setup script"
Assert-File $iconGenerator "Icon generator"
Assert-File $signatureVerifier "Authenticode signing helper"
Assert-File (Join-Path $projectRoot "LICENSE") "Project AGPL license"

$gitHead = (& git -C $projectRoot rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $gitHead -notmatch '^[0-9a-f]{40}$') {
    throw "Could not determine the repository HEAD revision"
}
$gitStatus = @(& git -C $projectRoot status --porcelain=v1 --untracked-files=all)
$isDirty = $gitStatus.Count -gt 0
$isSignedBuild = -not [string]::IsNullOrWhiteSpace($PeSignCommand)
$hasInnoSigning = -not [string]::IsNullOrWhiteSpace($SignToolCommand)

if ([string]::IsNullOrWhiteSpace($SignToolCommand) -xor [string]::IsNullOrWhiteSpace($SignToolName)) {
    throw "SignToolName and SignToolCommand must be supplied together"
}
if ($isSignedBuild -xor $hasInnoSigning) {
    throw "Project executable and Inno installer/uninstaller signing must be configured together"
}
if ($BuildMode -eq "Release") {
    if (-not $isSignedBuild -or [string]::IsNullOrWhiteSpace($SignToolName)) {
        throw "Release mode requires trusted signing for the project executable, installer, and uninstaller"
    }
    Assert-File $SignToolPath "Microsoft signtool"
    Assert-File $PeSignCommand "Project PE signing command"
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
}
else {
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

$dumpbin = Join-Path $ToolRoot "vs\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\dumpbin.exe"
$iscc = Join-Path $ToolRoot "inno-setup\ISCC.exe"
$openCvBin = Join-Path $ToolRoot "opencv\opencv\build\x64\vc16\bin"
$cudaBin = Join-Path $ToolRoot "cuda-runtime\nvidia\cuda_runtime\bin"
$tensorRtRoot = Join-Path $ToolRoot "tensorrt\TensorRT-11.1.0.106"
$tensorRtBin = Join-Path $tensorRtRoot "bin"
$msvcCrt = Join-Path $ToolRoot "vs\VC\Redist\MSVC\14.44.35112\x64\Microsoft.VC143.CRT"
$searchDirectories = @($openCvBin, $cudaBin, $tensorRtBin, $msvcCrt)

Assert-File $dumpbin "MSVC dumpbin"
Assert-File $iscc "Inno Setup compiler"
foreach ($directory in $searchDirectories) {
    Assert-Directory $directory "Runtime dependency directory"
}

$sourceInputs = @(
    (Join-Path $projectRoot "native\CMakeLists.txt")
    (Get-ChildItem -LiteralPath (Join-Path $projectRoot "native\src") -File -Filter "*.cpp").FullName
    (Get-ChildItem -LiteralPath (Join-Path $projectRoot "native\include") -Recurse -File -Include "*.hpp", "*.h").FullName
)
$newerSource = $sourceInputs | Where-Object {
    (Get-Item -LiteralPath $_).LastWriteTimeUtc -gt (Get-Item -LiteralPath $ReleaseExecutable).LastWriteTimeUtc
}
if ($newerSource) {
    throw "The Release executable is older than native source input: $($newerSource -join ', ')"
}

if ($isSignedBuild) {
    Assert-File $SignToolPath "Microsoft signtool"
    Assert-File $PeSignCommand "Project PE signing command"
    & $PeSignCommand -FilePath $ReleaseExecutable
    if (-not $?) {
        throw "Project PE signing command failed"
    }
    & $signatureVerifier -FilePath $ReleaseExecutable -SignToolPath $SignToolPath -VerifyOnly
}
elseif ((Get-AuthenticodeSignature -LiteralPath $ReleaseExecutable).Status -ne
    [System.Management.Automation.SignatureStatus]::NotSigned) {
    throw "Explicit unsigned preview expected the project executable to be NotSigned"
}

$stageParent = Split-Path -Parent $StageDir
$outputParent = Split-Path -Parent $OutputDir
foreach ($parent in @($stageParent, $outputParent)) {
    if (-not (Test-Path -LiteralPath $parent)) {
        $grandParent = Split-Path -Parent $parent
        Assert-Directory $grandParent "Generated directory parent"
        New-Item -ItemType Directory -Path $parent | Out-Null
    }
}
Reset-Directory $StageDir
if (-not (Test-Path -LiteralPath $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
}

$stageBin = New-Item -ItemType Directory -Path (Join-Path $StageDir "bin")
$stageDocs = New-Item -ItemType Directory -Path (Join-Path $StageDir "docs")
$stageLicenses = New-Item -ItemType Directory -Path (Join-Path $StageDir "licenses")

$resolved = [ordered]@{}
$imports = [System.Collections.Generic.List[object]]::new()
$queue = [System.Collections.Generic.Queue[string]]::new()

function Add-StagedBinary([string]$Source, [string]$Reason) {
    $name = Split-Path -Leaf $Source
    if ($resolved.Contains($name)) {
        return
    }
    Assert-X64Pe $Source $dumpbin
    Copy-Item -LiteralPath $Source -Destination (Join-Path $stageBin.FullName $name)
    $resolved[$name] = [pscustomobject]@{
        source = $Source
        reason = $Reason
        sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $Source).Hash
    }
    $queue.Enqueue($Source)
}

Add-StagedBinary $ReleaseExecutable "Application executable"
$ffmpegPlugin = Join-Path $openCvBin "opencv_videoio_ffmpeg4120_64.dll"
Assert-File $ffmpegPlugin "OpenCV FFmpeg videoio plugin"
$ffmpegSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $ffmpegPlugin).Hash.ToLowerInvariant()
$ffmpegMd5 = (Get-FileHash -Algorithm MD5 -LiteralPath $ffmpegPlugin).Hash.ToLowerInvariant()
if ($ffmpegSha256 -cne "a0f01e4ee5e97b4a513cd70f01fafadc0dd187ba5d1293cb7fc6b77e7d17c631" -or
    $ffmpegMd5 -cne "e5c6936240201064b15bcecf1816e8f4") {
    throw "OpenCV FFmpeg plugin does not match the documented OpenCV 4.12.0 binary"
}
Add-StagedBinary $ffmpegPlugin "OpenCV videoio plugin loaded by filename for RTSP and offline capture"

$systemDirectory = [Environment]::SystemDirectory
while ($queue.Count -gt 0) {
    $binary = $queue.Dequeue()
    foreach ($dependency in Get-PeDependencies $binary $dumpbin) {
        $resolvedPath = Find-Dependency $dependency $searchDirectories
        if ($resolvedPath) {
            Add-StagedBinary $resolvedPath "PE import required by $(Split-Path -Leaf $binary)"
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

$unexpectedTensorRt = @($resolved.Keys | Where-Object {
    $_ -match '^(nvonnxparser|nvinfer_plugin|nvinfer_builder|nvinfer_vc_plugin)'
})
if ($unexpectedTensorRt) {
    throw "Development/plugin TensorRT DLLs were staged without an approved runtime boundary: $($unexpectedTensorRt -join ', ')"
}

Copy-Item -LiteralPath (Join-Path $scriptRoot "README.md") -Destination $stageDocs.FullName
Copy-Item -LiteralPath (Join-Path $scriptRoot "THIRD_PARTY_NOTICES.md") -Destination $stageDocs.FullName
Copy-Item -LiteralPath (Join-Path $scriptRoot "FFMPEG-SOURCE.md") -Destination $stageDocs.FullName
Copy-Item -LiteralPath (Join-Path $projectRoot "LICENSE") -Destination (Join-Path $stageLicenses.FullName "AGPL-3.0.txt")
Copy-Item -LiteralPath (Join-Path $projectRoot "LICENSES.md") -Destination $stageDocs.FullName
Copy-Item -LiteralPath (Join-Path $projectRoot "SECURITY.md") -Destination $stageDocs.FullName
Copy-Item -LiteralPath (Join-Path $scriptRoot "licenses") -Destination (Join-Path $stageLicenses.FullName "FFmpeg-dependencies") -Recurse
Copy-Item -LiteralPath (Join-Path $ToolRoot "opencv\opencv\LICENSE.txt") -Destination (Join-Path $stageLicenses.FullName "OpenCV-LICENSE.txt")
Copy-Item -LiteralPath (Join-Path $ToolRoot "opencv\opencv\LICENSE_FFMPEG.txt") -Destination (Join-Path $stageLicenses.FullName "OpenCV-FFmpeg-LGPL-2.1.txt")
$openCvNoticeDir = Join-Path $ToolRoot "opencv\opencv\build\etc\licenses"
Assert-Directory $openCvNoticeDir "OpenCV bundled third-party notices"
Copy-Item -LiteralPath $openCvNoticeDir -Destination (Join-Path $stageLicenses.FullName "OpenCV-third-party") -Recurse
$cudaLicense = Join-Path $ToolRoot "cuda-runtime\nvidia_cuda_runtime_cu12-12.9.79.dist-info\licenses\License.txt"
Assert-File $cudaLicense "CUDA runtime license"
Copy-Item -LiteralPath $cudaLicense -Destination (Join-Path $stageLicenses.FullName "NVIDIA-CUDA-License.txt")
Copy-Item -LiteralPath (Join-Path $tensorRtRoot "doc\README.txt") -Destination (Join-Path $stageLicenses.FullName "NVIDIA-TensorRT-README.txt")

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
@"
Cuajone PPE Monitor source offer and release correspondence

Application version: $Version
Numeric file version: $FileVersion
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
and retained for the applicable period. Upstream links alone are not treated as a
substitute for every AGPL or LGPL source-conveyance obligation. See
docs/FFMPEG-SOURCE.md and docs/THIRD_PARTY_NOTICES.md.
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

$iconPath = Join-Path $StageDir "CuajonePPEMonitor.ico"
& $iconGenerator -OutputPath $iconPath
if ($LASTEXITCODE -ne 0) {
    throw "Icon generation failed"
}
Assert-File $iconPath "Generated application icon"

$metadata = [ordered]@{
    product = "Cuajone PPE Monitor"
    appVersion = $Version
    fileVersion = $FileVersion
    architecture = "x64"
    configuration = "Release"
    buildUtc = [DateTime]::UtcNow.ToString("o")
    releaseExecutable = $ReleaseExecutable
    releaseExecutableSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $ReleaseExecutable).Hash
    toolRoot = $ToolRoot
    dumpbin = $dumpbin
    innoSetup = "7.0.2 x64"
    licenseStatus = "Original project source is AGPL-3.0-only; third-party artifacts, models, and datasets retain separate upstream terms"
    releaseStatus = if ($BuildMode -eq "Release") {
        "Open-source release with exact source archives and trusted Authenticode required"
    } else {
        "Explicit internal preview; unsigned output permitted only by switch; real-engine operation not validated"
    }
    buildMode = $BuildMode
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
    projectLicenseSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $projectRoot "LICENSE")).Hash.ToLowerInvariant()
    signingPolicy = if ($isSignedBuild) { "Trusted Authenticode required and verified" } else { "Explicit unsigned internal preview" }
    thirdPartyBinariesResigned = $false
    acceptanceScope = "Loader and --help only; no engines, cameras, preflight, or inference"
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

Assert-NoForbiddenFiles $StageDir
$manifestPath = Join-Path $StageDir "SHA256SUMS.txt"
$manifestLines = foreach ($file in Get-ChildItem -LiteralPath $StageDir -Recurse -File | Sort-Object FullName) {
    if ($file.FullName -eq $manifestPath) {
        continue
    }
    $relative = [System.IO.Path]::GetRelativePath($StageDir, $file.FullName).Replace('\', '/')
    "{0}  {1}" -f (Get-FileHash -Algorithm SHA256 -LiteralPath $file.FullName).Hash.ToLowerInvariant(), $relative
}
$manifestLines | Set-Content -LiteralPath $manifestPath -Encoding ASCII
Assert-NoForbiddenFiles $StageDir

$outputBaseFilename = if ($BuildMode -eq "Preview") {
    "CuajonePPEMonitor-$Version-x64-Internal-Setup"
} else {
    "CuajonePPEMonitor-$Version-x64-Setup"
}
$setupOriginalFilename = "CuajonePPEMonitorSetup.exe"
$compilerOutputBaseFilename = [System.IO.Path]::GetFileNameWithoutExtension(
    $setupOriginalFilename
)
$isccArguments = @(
    "/DStageDir=$StageDir",
    "/DOutputDir=$OutputDir",
    "/DAppVersion=$Version",
    "/DFileVersion=$FileVersion",
    "/DOutputBaseFilename=$compilerOutputBaseFilename",
    "/DSetupOriginalFilename=$setupOriginalFilename"
)
if ($BuildMode -eq "Preview") {
    $isccArguments += "/DPreviewBuild"
}
if (-not [string]::IsNullOrWhiteSpace($SignToolCommand)) {
    $isccArguments += "/S$SignToolName=$SignToolCommand"
    $isccArguments += "/DSignToolName=$SignToolName"
}
$isccArguments += $installerScript

& $iscc @isccArguments
if ($LASTEXITCODE -ne 0) {
    throw "Inno Setup compilation failed with exit code $LASTEXITCODE"
}

$compiledInstallerPath = Join-Path $OutputDir "$compilerOutputBaseFilename.exe"
Assert-File $compiledInstallerPath "Compiled installer"
$installerPath = Join-Path $OutputDir "$outputBaseFilename.exe"
if (Test-Path -LiteralPath $installerPath) {
    Remove-Item -LiteralPath $installerPath -Force
}
Move-Item -LiteralPath $compiledInstallerPath -Destination $installerPath
Assert-X64Pe $installerPath $dumpbin
$installerHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $installerPath).Hash.ToLowerInvariant()
$sidecarPath = "$installerPath.sha256"
"$installerHash  $(Split-Path -Leaf $installerPath)" | Set-Content -LiteralPath $sidecarPath -Encoding ASCII
$signature = Get-AuthenticodeSignature -LiteralPath $installerPath
if ($isSignedBuild) {
    & $signatureVerifier -FilePath $installerPath -SignToolPath $SignToolPath -VerifyOnly
}
elseif ($signature.Status -ne [System.Management.Automation.SignatureStatus]::NotSigned) {
    throw "Explicit unsigned preview expected the installer to be NotSigned"
}
$versionInfo = (Get-Item -LiteralPath $installerPath).VersionInfo
$originalFilename = $versionInfo.OriginalFilename.TrimEnd()
if ($originalFilename -cne $setupOriginalFilename) {
    throw "Installer OriginalFilename mismatch: expected '$setupOriginalFilename', got '$originalFilename'"
}

[pscustomobject]@{
    Installer = $installerPath
    SHA256 = $installerHash
    SizeBytes = (Get-Item -LiteralPath $installerPath).Length
    SignatureStatus = $signature.Status
    FileVersion = $versionInfo.FileVersion.TrimEnd()
    ProductVersion = $versionInfo.ProductVersion.TrimEnd()
    OriginalFilename = $originalFilename
    Stage = $StageDir
    Manifest = $manifestPath
    Sidecar = $sidecarPath
    StagedBinaries = @($resolved.Keys)
}
