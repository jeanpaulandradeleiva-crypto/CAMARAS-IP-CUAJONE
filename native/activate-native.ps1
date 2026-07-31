# SPDX-License-Identifier: AGPL-3.0-only

[CmdletBinding()]
param(
    [switch]$CpuOnly
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$toolRoot = Join-Path $projectRoot ".tools\native"
$vsRoot = "$toolRoot\vs"
$cmakeBin = "$toolRoot\cmake\cmake-3.31.8-windows-x86_64\bin"
$ninjaBin = "$toolRoot\ninja"
$cudaRoot = "$toolRoot\cuda-runtime\nvidia\cuda_runtime"
$cudaCompilerHeadersRoot = "$toolRoot\cuda-nvcc\nvidia\cuda_nvcc"
$cudaCcclHeadersRoot = "$toolRoot\cuda-cccl\nvidia\cuda_cccl"
$tensorRtRoot = "$toolRoot\tensorrt\TensorRT-11.1.0.106"
$onnxRuntimeRoot = "$toolRoot\onnxruntime-win-x64-1.25.0"
$openCvRoot = "$toolRoot\opencv\opencv\build"
$openCvLib = "$openCvRoot\x64\vc16\lib"
$openCvBin = "$openCvRoot\x64\vc16\bin"
$tempRoot = "$toolRoot\temp"
$vsDevCmd = "$vsRoot\Common7\Tools\VsDevCmd.bat"

$requiredPaths = @(
    $vsDevCmd,
    "$cmakeBin\cmake.exe",
    "$ninjaBin\ninja.exe",
    "$onnxRuntimeRoot\include\onnxruntime_cxx_api.h",
    "$onnxRuntimeRoot\lib\onnxruntime.lib",
    "$onnxRuntimeRoot\lib\onnxruntime.dll",
    "$openCvLib\OpenCVConfig.cmake"
)
if (-not $CpuOnly) {
    $requiredPaths += @(
        "$cudaRoot\include\cuda_runtime_api.h",
        "$cudaRoot\lib\x64\cudart.lib",
        "$cudaCompilerHeadersRoot\include\crt\host_defines.h",
        "$cudaCcclHeadersRoot\include\nv\target",
        "$tensorRtRoot\include\NvInfer.h",
        "$tensorRtRoot\lib\nvinfer_11.lib"
    )
}
foreach ($requiredPath in $requiredPaths) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Required native toolchain path is missing: $requiredPath"
    }
}

$env:TEMP = $tempRoot
$env:TMP = $tempRoot
$env:ONNXRUNTIME_ROOT = $onnxRuntimeRoot
$env:OpenCV_DIR = $openCvLib
if ($CpuOnly) {
    $env:TENSORRT_ROOT = $null
    $env:CUDA_RUNTIME_ROOT = $null
    $env:CUDA_COMPILER_HEADERS_ROOT = $null
    $env:CUDA_CCCL_HEADERS_ROOT = $null
    $env:PATH = "$cmakeBin;$ninjaBin;$onnxRuntimeRoot\lib;$openCvBin;$env:PATH"
} else {
    $env:TENSORRT_ROOT = $tensorRtRoot
    $env:CUDA_RUNTIME_ROOT = $cudaRoot
    $env:CUDA_COMPILER_HEADERS_ROOT = $cudaCompilerHeadersRoot
    $env:CUDA_CCCL_HEADERS_ROOT = $cudaCcclHeadersRoot
    $env:PATH = "$cmakeBin;$ninjaBin;$cudaRoot\bin;$tensorRtRoot\bin;$onnxRuntimeRoot\lib;$openCvBin;$env:PATH"
}

$developerEnvironment = @(
    & $env:ComSpec /d /s /c "call `"$vsDevCmd`" -arch=amd64 -host_arch=amd64 >nul && set"
)
if ($LASTEXITCODE -ne 0) {
    throw "Could not activate the local Visual Studio developer environment"
}
foreach ($entry in $developerEnvironment) {
    $separator = $entry.IndexOf('=')
    if ($separator -gt 0) {
        Set-Item -LiteralPath "Env:$($entry.Substring(0, $separator))" -Value $entry.Substring($separator + 1)
    }
}

Write-Host "Cuajone native MSVC environment activated from $toolRoot (CPU only: $CpuOnly)"
