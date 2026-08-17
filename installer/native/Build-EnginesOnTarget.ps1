# SPDX-License-Identifier: AGPL-3.0-only

[CmdletBinding()]
param(
    [string]$PpeOnnx,

    [string]$PoseOnnx,

    [Parameter(Mandatory = $true)]
    [string]$OutputDir,

    [string]$TensorRtBin,

    [ValidateNotNullOrEmpty()]
    [string]$PpeLabels = "Gloves,Person,Safety_boots,Vest,respirador,tapaorejas,Hard_hat,lentes_protectores",

    [ValidateRange(1, 1440)]
    [int]$TimeoutMinutes = 45
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Resolve self-relative defaults so the MSI custom action only needs -OutputDir:
# the script, ONNX models, and trtexec all live together in bin\engine-builder.
if ([string]::IsNullOrWhiteSpace($TensorRtBin)) { $TensorRtBin = $PSScriptRoot }
if ([string]::IsNullOrWhiteSpace($PpeOnnx)) { $PpeOnnx = Join-Path $PSScriptRoot "ppe.onnx" }
if ([string]::IsNullOrWhiteSpace($PoseOnnx)) { $PoseOnnx = Join-Path $PSScriptRoot "pose.onnx" }

# MSI directory properties (e.g. [INSTALLFOLDER]) carry a trailing backslash, which
# yields doubled separators after Join-Path. Normalize them for clean trtexec paths.
$TensorRtBin = $TensorRtBin.TrimEnd('\')
$OutputDir = $OutputDir.TrimEnd('\')

function Assert-File([string]$Path, [string]$Description) {
    if ([string]::IsNullOrWhiteSpace($Path) -or -not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Description was not found: $Path"
    }
}

function Assert-Directory([string]$Path, [string]$Description) {
    if ([string]::IsNullOrWhiteSpace($Path) -or -not (Test-Path -LiteralPath $Path -PathType Container)) {
        throw "$Description was not found: $Path"
    }
}

function Write-Timestamped([string]$Message) {
    Write-Host ("[{0}] {1}" -f ([DateTime]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ")), $Message)
}

function Resolve-NvidiaSmi {
    $command = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }
    $systemPath = Join-Path $env:ProgramFiles "NVIDIA Corporation\NVSMI\nvidia-smi.exe"
    if (Test-Path -LiteralPath $systemPath -PathType Leaf) {
        return $systemPath
    }
    throw "nvidia-smi was not found. A compatible NVIDIA GPU and driver are required to build TensorRT engines on this machine."
}

function Invoke-TrtexecBuild(
    [string]$Onnx,
    [string]$Engine,
    [string]$Description,
    [string]$Trtexec,
    [string]$WorkingDirectory
) {
    $arguments = @(
        ('--onnx="' + $Onnx + '"'),
        ('--saveEngine="' + $Engine + '"'),
        "--minShapes=images:1x3x640x640",
        "--optShapes=images:1x3x640x640",
        "--maxShapes=images:1x3x1280x1280"
    )

    $process = [System.Diagnostics.Process]::new()
    $process.StartInfo.FileName = $Trtexec
    $process.StartInfo.WorkingDirectory = $WorkingDirectory
    $process.StartInfo.Arguments = $arguments -join " "
    $process.StartInfo.UseShellExecute = $false
    $process.StartInfo.CreateNoWindow = $true

    Write-Timestamped "Building $Description (timeout: $TimeoutMinutes minutes)"
    Write-Host "  trtexec $($arguments -join ' ')"
    if (-not $process.Start()) {
        throw "Failed to start trtexec for $Description"
    }

    $timeoutMs = $TimeoutMinutes * 60 * 1000
    $completed = $process.WaitForExit($timeoutMs)
    if (-not $completed) {
        try {
            $process.Kill()
        } catch {
        }
        throw "$Description timed out after $TimeoutMinutes minutes and was terminated. No engine was produced."
    }
    $process.WaitForExit()
    if ($process.ExitCode -ne 0) {
        throw "$Description failed with trtexec exit code $($process.ExitCode). See the trtexec output above."
    }
    $process.Dispose()
    Assert-File $Engine "$Description engine"
    Write-Timestamped "Built $Description -> $Engine"
}

try {
    Assert-Directory $TensorRtBin "TensorRT bin directory"
    $trtexec = Join-Path $TensorRtBin "trtexec.exe"
    Assert-File $trtexec "TensorRT trtexec executable"

    Assert-File $PpeOnnx "PPE ONNX model"
    Assert-File $PoseOnnx "Pose ONNX model"

    if (-not (Test-Path -LiteralPath $OutputDir -PathType Container)) {
        New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
    }
    Assert-Directory $OutputDir "Engine output directory"

    $nvidiaSmi = Resolve-NvidiaSmi
    $gpuOutput = @(& $nvidiaSmi --query-gpu=name,compute_cap --format=csv,noheader,nounits 2>&1)
    if ($LASTEXITCODE -ne 0 -or $gpuOutput.Count -eq 0) {
        throw "nvidia-smi could not report an NVIDIA GPU. Verify the driver installation.`n$($gpuOutput -join [Environment]::NewLine)"
    }
    $gpuLine = ($gpuOutput | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | Select-Object -First 1).Trim()
    $gpuFields = $gpuLine -split ","
    if ($gpuFields.Count -lt 2) {
        throw "Unexpected nvidia-smi output; expected 'name, compute_cap': $gpuLine"
    }
    $gpuName = $gpuFields[0].Trim()
    $computeCapability = $gpuFields[1].Trim()
    if ($gpuOutput.Count -gt 1) {
        Write-Warning "Multiple GPUs detected; building for the first GPU ($gpuName, SM $computeCapability)"
    }
    if ($computeCapability -match '^\s*(\d+)\.(\d+)\s*$') {
        $sm = ([int]$Matches[1]) * 10 + [int]$Matches[2]
    } elseif ($computeCapability -match '^\s*(\d+)\s*$') {
        $sm = [int]$Matches[1]
    } else {
        throw "Could not parse the NVIDIA compute capability '$computeCapability'"
    }

    Write-Timestamped "Resolved GPU: $gpuName (compute capability $computeCapability, SM $sm)"
    Write-Timestamped "PPE labels: $PpeLabels"

    $ppeEnginePath = Join-Path $OutputDir "ppe.engine"
    $poseEnginePath = Join-Path $OutputDir "pose.engine"

    Write-Progress -Id 1 -Activity "Building TensorRT engines" -Status "Compiling PPE engine (1 of 2)..." -PercentComplete 0
    Invoke-TrtexecBuild $PpeOnnx $ppeEnginePath "PPE engine" $trtexec $TensorRtBin

    Write-Progress -Id 1 -Activity "Building TensorRT engines" -Status "Compiling pose engine (2 of 2)..." -PercentComplete 50
    Invoke-TrtexecBuild $PoseOnnx $poseEnginePath "pose engine" $trtexec $TensorRtBin

    Assert-File $ppeEnginePath "PPE engine"
    Assert-File $poseEnginePath "Pose engine"
    $ppeEngineSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $ppeEnginePath).Hash.ToLowerInvariant()
    $poseEngineSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $poseEnginePath).Hash.ToLowerInvariant()

    $sumsPath = Join-Path $OutputDir "SHA256SUMS.txt"
    @(
        "# NexoAI Vision TensorRT engine build"
        "# GPU: $gpuName (compute capability $computeCapability, SM $sm)"
        "# Generated: $([DateTime]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ"))"
        "$ppeEngineSha256  ppe.engine"
        "$poseEngineSha256  pose.engine"
    ) | Set-Content -LiteralPath $sumsPath -Encoding ASCII

    $manifestPath = Join-Path $OutputDir "engine-build-manifest.json"
    [ordered]@{
        generated_utc = [DateTime]::UtcNow.ToString("o")
        gpu_name = $gpuName
        gpu_compute_capability = $computeCapability
        gpu_sm = $sm
        ppe_labels = $PpeLabels
        ppe_engine_file = "ppe.engine"
        ppe_engine_sha256 = $ppeEngineSha256
        pose_engine_file = "pose.engine"
        pose_engine_sha256 = $poseEngineSha256
        trtexec = $trtexec
        tensor_rt_bin = $TensorRtBin
    } | ConvertTo-Json | Set-Content -LiteralPath $manifestPath -Encoding UTF8

    Write-Progress -Id 1 -Activity "Building TensorRT engines" -Status "Complete" -Completed
    Write-Timestamped "Engine build complete: ppe.engine and pose.engine written to $OutputDir"
    Write-Host "Build manifest: $manifestPath"
} catch {
    Write-Error "Engine build failed: $($_.Exception.Message)"
    exit 1
}
