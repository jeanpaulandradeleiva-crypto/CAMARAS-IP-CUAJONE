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
    [string]$WixToolRoot = "D:\DevTools\CuajoneNative\wix",
    [string]$WixVersion = "6.0.2",
    [string]$ReleaseExecutable = "D:\DevTools\CuajoneNative\build\windows-msvc\Release\cuajone_native.exe",
    [string]$StageDir = "D:\DevTools\CuajoneNative\installer\stage",
    [string]$WixBuildDir = "D:\DevTools\CuajoneNative\installer\wix-build",
    [string]$OutputDir = "D:\DevTools\CuajoneNative\installer\output",
    [string]$VerificationRoot = "D:\DevTools\CuajoneNative\installer\msi-verification",
    [string]$SourceRevision = $env:CUAJONE_SOURCE_REVISION,
    [string]$SourceArchiveUrl = $env:CUAJONE_SOURCE_ARCHIVE_URL,
    [string]$SourceArchiveSha256 = $env:CUAJONE_SOURCE_ARCHIVE_SHA256,
    [string]$FfmpegSourceArchiveUrl = $env:CUAJONE_FFMPEG_SOURCE_ARCHIVE_URL,
    [string]$FfmpegSourceArchiveSha256 = $env:CUAJONE_FFMPEG_SOURCE_ARCHIVE_SHA256,
    [string]$SignToolPath = $env:CUAJONE_SIGNTOOL_PATH,
    [string]$SignCommand = $env:CUAJONE_SIGN_COMMAND
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptRoot = $PSScriptRoot
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $scriptRoot "..\..")).Path
$packageSource = Join-Path $scriptRoot "Package.wxs"
$packageProject = Join-Path $scriptRoot "CuajonePpeMonitor.wixproj"
$iconGenerator = Join-Path $scriptRoot "generate-icon.ps1"
$signatureVerifier = Join-Path $scriptRoot "sign-release.ps1"
$packageVerifier = Join-Path $scriptRoot "test-installer.ps1"
$sourceRepository = "https://github.com/jeanpaulandradeleiva-crypto/CAMARAS-IP-CUAJONE"
$upgradeCode = "88A886C2-8F6D-4669-B6FB-7DFC1E7B0397"

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

function Ensure-Directory([string]$Path) {
    Assert-DDrivePath $Path "Generated directory"
    if (Test-Path -LiteralPath $Path) {
        Assert-Directory $Path "Generated directory"
        return
    }
    $parent = Split-Path -Parent $Path
    Assert-Directory $parent "Generated directory parent"
    New-Item -ItemType Directory -Path $Path | Out-Null
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

function Write-PayloadSource([string]$Root, [string]$OutputPath) {
    $lines = [System.Collections.Generic.List[string]]::new()
    $lines.Add('<!-- Generated by build-installer.ps1. Do not commit. -->')
    $lines.Add('<Wix xmlns="http://wixtoolset.org/schemas/v4/wxs">')
    $lines.Add('  <Fragment>')
    $lines.Add('    <ComponentGroup Id="PayloadComponents" Directory="INSTALLFOLDER">')

    foreach ($file in Get-ChildItem -LiteralPath $Root -Recurse -File | Sort-Object FullName) {
        $relative = [System.IO.Path]::GetRelativePath($Root, $file.FullName).Replace('/', '\')
        $hex = Get-StableHex $relative
        $componentId = "Component_$($hex.Substring(0, 24))"
        $fileId = switch -CaseSensitive ($relative) {
            'bin\cuajone_native.exe' { 'AppExecutable'; break }
            'docs\README.md' { 'DeploymentReadme'; break }
            'CuajonePPEMonitor.ico' { 'InstalledProductIcon'; break }
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
        $lines.Add("      <Component Id=`"$componentId`" Guid=`"$guid`"$subdirectoryAttribute>")
        if ($fileId -ceq 'AppExecutable') {
            $lines.Add("        <File Id=`"$fileId`" Source=`"$source`" KeyPath=`"yes`">")
            $lines.Add('          <Shortcut Id="CommandHelpShortcut" Directory="ProgramMenuAppFolder" Name="Cuajone PPE Monitor - Command Help" Description="Open command-line help" Arguments="--help" WorkingDirectory="INSTALLFOLDER" Advertise="yes" />')
            $lines.Add('        </File>')
            $lines.Add('        <RemoveFolder Id="RemoveProgramMenuAppFolder" Directory="ProgramMenuAppFolder" On="uninstall" />')
        } elseif ($fileId -ceq 'DeploymentReadme') {
            $lines.Add("        <File Id=`"$fileId`" Source=`"$source`" KeyPath=`"yes`">")
            $lines.Add('          <Shortcut Id="ReadmeShortcut" Directory="ProgramMenuAppFolder" Name="Cuajone PPE Monitor - README" Description="Open deployment and license documentation" WorkingDirectory="INSTALLFOLDER" Advertise="yes" />')
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

foreach ($path in @($StageDir, $WixBuildDir, $OutputDir, $VerificationRoot, $WixToolRoot)) {
    Assert-DDrivePath $path "Build path"
}
Assert-Directory $ToolRoot "Native tool root"
Assert-Directory $WixToolRoot "WiX tool root"
Assert-File $ReleaseExecutable "Release executable"
Assert-File $packageSource "WiX package source"
Assert-File $packageProject "WiX project"
Assert-File $iconGenerator "Icon generator"
Assert-File $signatureVerifier "Authenticode signing helper"
Assert-File $packageVerifier "MSI verification helper"
Assert-File (Join-Path $projectRoot "LICENSE") "Project AGPL license"

$gitHead = (& git -C $projectRoot rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $gitHead -notmatch '^[0-9a-f]{40}$') {
    throw "Could not determine the repository HEAD revision"
}
$gitStatus = @(& git -C $projectRoot status --porcelain=v1 --untracked-files=all)
$isDirty = $gitStatus.Count -gt 0
$isSignedBuild = -not [string]::IsNullOrWhiteSpace($SignCommand)
$isInternalPilotSigning = $env:CUAJONE_ALLOW_INTERNAL_PILOT_TRUST -ceq "1"

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
$tensorRtRoot = Join-Path $ToolRoot "tensorrt\TensorRT-11.1.0.106"
$tensorRtBin = Join-Path $tensorRtRoot "bin"
$msvcCrt = Join-Path $ToolRoot "vs\VC\Redist\MSVC\14.44.35112\x64\Microsoft.VC143.CRT"
$searchDirectories = @($openCvBin, $cudaBin, $tensorRtBin, $msvcCrt)

Assert-File $dumpbin "MSVC dumpbin"
Assert-File $wix "WiX CLI"
foreach ($directory in $searchDirectories) {
    Assert-Directory $directory "Runtime dependency directory"
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
    Assert-File $SignCommand "Artifact signing command"
    & $SignCommand -FilePath $ReleaseExecutable
    if (-not $?) {
        throw "Project executable signing command failed"
    }
    & $signatureVerifier -FilePath $ReleaseExecutable -SignToolPath $SignToolPath -VerifyOnly
} elseif ((Get-AuthenticodeSignature -LiteralPath $ReleaseExecutable).Status -ne
    [System.Management.Automation.SignatureStatus]::NotSigned) {
    throw "Explicit unsigned preview expected the project executable to be NotSigned"
}

$generatedParent = Split-Path -Parent $StageDir
Ensure-Directory $generatedParent
foreach ($path in @($WixBuildDir, $OutputDir, $VerificationRoot)) {
    $parent = Split-Path -Parent $path
    Ensure-Directory $parent
}
Reset-Directory $StageDir
Reset-Directory $WixBuildDir
Ensure-Directory $OutputDir
Ensure-Directory $VerificationRoot

$stageBin = New-Item -ItemType Directory -Path (Join-Path $StageDir "bin")
$stageDocs = New-Item -ItemType Directory -Path (Join-Path $StageDir "docs")
$stageLicenses = New-Item -ItemType Directory -Path (Join-Path $StageDir "licenses")

$resolved = [ordered]@{}
$imports = [System.Collections.Generic.List[object]]::new()
$stagedSources = [System.Collections.Generic.List[object]]::new()
$queue = [System.Collections.Generic.Queue[string]]::new()
$sourceRoots = [ordered]@{
    repository = $projectRoot
    build = (Join-Path $ToolRoot "build")
    tool = $ToolRoot
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
    $destinationParent = Split-Path -Parent $resolvedDestination
    if (-not (Test-Path -LiteralPath $destinationParent -PathType Container)) {
        New-Item -ItemType Directory -Path $destinationParent -Force | Out-Null
    }

    Copy-Item -LiteralPath $resolvedSource -Destination $resolvedDestination
    $sourceHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $resolvedSource).Hash.ToLowerInvariant()
    $stagedHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $resolvedDestination).Hash.ToLowerInvariant()
    if ($stagedHash -cne $sourceHash) {
        throw "Staging changed source bytes: $resolvedSource"
    }

    $stagedSources.Add([ordered]@{
        sourceScope = $SourceScope
        sourcePath = $resolvedSource
        sourceRelativePath = [System.IO.Path]::GetRelativePath($scopeRoot, $resolvedSource).Replace('\', '/')
        stagedRelativePath = [System.IO.Path]::GetRelativePath($resolvedStage, $resolvedDestination).Replace('\', '/')
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
Copy-StagedInput (Join-Path $tensorRtRoot "doc\README.txt") (Join-Path $stageLicenses.FullName "NVIDIA-TensorRT-README.txt") "tool" "TensorRT redistribution and license reference"

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
    msiVersion = $msiVersion
    architecture = "x64"
    configuration = "Release"
    buildUtc = [DateTime]::UtcNow.ToString("o")
    releaseExecutable = $ReleaseExecutable
    releaseExecutableSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $ReleaseExecutable).Hash
    toolRoot = $ToolRoot
    dumpbin = $dumpbin
    wixToolset = $wixReportedVersion
    upgradeCode = $upgradeCode
    installFolderDefault = "C:\Program Files\Cuajone PPE Monitor"
    dataFolder = "C:\ProgramData\Cuajone PPE Monitor\runtime"
    licenseStatus = "Original project source is AGPL-3.0-only; third-party artifacts, models, and datasets retain separate upstream terms"
    releaseStatus = if ($BuildMode -eq "Release") {
        "Open-source release with exact source archives and trusted Authenticode required"
    } else {
        "Internal preview; public trust is not implied; real-engine operation not validated"
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
    signingPolicy = if ($isInternalPilotSigning) {
        "Private Authenticode for enrolled internal pilot machines"
    } elseif ($isSignedBuild) {
        "Publicly trusted Authenticode required and verified"
    } else {
        "Explicit unsigned internal preview"
    }
    thirdPartyBinariesResigned = $false
    acceptanceScope = "MSI database, administrative extraction, loader, and --help only; no install, engines, cameras, preflight, or inference"
    stagingProvenanceVersion = 1
    sourceRoots = $sourceRoots
    sourceProvenance = @($stagedSources)
    generatedStagePaths = @(
        "CuajonePPEMonitor.ico"
        "build-metadata.json"
        "docs/SOURCE-OFFER.txt"
        "licenses/Microsoft-VC-Runtime-REDISTRIBUTION-REFERENCE.txt"
        "licenses/NVIDIA-TensorRT-LICENSE-REFERENCE.txt"
        "SHA256SUMS.txt"
    )
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

$licenseText = Get-Content -LiteralPath (Join-Path $projectRoot "LICENSE") -Raw
$rtfText = $licenseText.Replace('\', '\\').Replace('{', '\{').Replace('}', '\}')
$rtfText = $rtfText -replace "\r?\n", "\par`r`n"
$licenseRtf = Join-Path $WixBuildDir "AGPL-3.0.rtf"
"{\rtf1\ansi\deff0{\fonttbl{\f0 Courier New;}}\fs18 $rtfText}" |
    Set-Content -LiteralPath $licenseRtf -Encoding ASCII

$payloadSource = Join-Path $WixBuildDir "Payload.wxs"
Write-PayloadSource $StageDir $payloadSource
Assert-File $payloadSource "Generated WiX payload source"

$outputBaseFilename = if ($BuildMode -eq "Preview") {
    "CuajonePPEMonitor-$Version-x64-Internal"
} else {
    "CuajonePPEMonitor-$Version-x64"
}
$installerPath = Join-Path $OutputDir "$outputBaseFilename.msi"
$wixPdb = Join-Path $WixBuildDir "$outputBaseFilename.wixpdb"
$wixIntermediate = Join-Path $WixBuildDir "obj"
Ensure-Directory $wixIntermediate
if (Test-Path -LiteralPath $installerPath) {
    Remove-Item -LiteralPath $installerPath -Force
}

$arpComments = if ($BuildMode -eq "Preview") {
    "Open-source AGPL-3.0 internal preview. Pilot signing and real-engine validation limitations apply."
} else {
    "Open-source AGPL-3.0 release. Third-party components retain their own license terms."
}
$wixArguments = @(
    "build",
    $packageSource,
    $payloadSource,
    "-arch", "x64",
    "-d", "StageDir=$StageDir",
    "-d", "AppVersion=$Version",
    "-d", "MsiVersion=$msiVersion",
    "-d", "ArpComments=$arpComments",
    "-d", "LicenseRtf=$licenseRtf",
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

$verificationParameters = @{
    InstallerPath = $installerPath
    StageDir = $StageDir
    AcceptanceRoot = $VerificationRoot
    WixToolRoot = $WixToolRoot
    SignToolPath = $SignToolPath
    ExpectedSignatureStatus = if ($isSignedBuild) { "Signed" } else { "NotSigned" }
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
    StagedBinaries = @($resolved.Keys)
    Verification = $verification
}
