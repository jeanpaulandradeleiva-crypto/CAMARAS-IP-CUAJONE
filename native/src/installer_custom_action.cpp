// SPDX-License-Identifier: AGPL-3.0-only

#include "cuajone/compute.hpp"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <msiquery.h>

#include <string>
#include <stdexcept>

namespace {

void setProperty(MSIHANDLE installation, const wchar_t* name, const std::wstring& value) {
    if (MsiSetPropertyW(installation, name, value.c_str()) != ERROR_SUCCESS) {
        throw std::runtime_error("MsiSetPropertyW failed");
    }
}

std::wstring wide(std::string_view value) {
    if (value.empty()) return {};
    const int required = MultiByteToWideChar(
        CP_UTF8, MB_ERR_INVALID_CHARS, value.data(), static_cast<int>(value.size()), nullptr, 0);
    if (required <= 0) return L"probe_error";
    std::wstring result(static_cast<std::size_t>(required), L'\0');
    if (MultiByteToWideChar(
            CP_UTF8, MB_ERR_INVALID_CHARS, value.data(), static_cast<int>(value.size()),
            result.data(), required) == 0) {
        return L"probe_error";
    }
    return result;
}

}  // namespace

extern "C" __declspec(dllexport) UINT __stdcall DetectComputeHardware(MSIHANDLE installation) noexcept {
    try {
        const cuajone::HardwareProbeResult probe = cuajone::probeHardware();
        setProperty(
            installation,
            L"NVIDIA_STATUS",
            wide(cuajone::hardwareProbeStatusName(probe.status)));
        setProperty(
            installation,
            L"CUDA_READY",
            probe.status == cuajone::HardwareProbeStatus::CudaReady ? L"1" : L"0");
        return ERROR_SUCCESS;
    } catch (...) {
        MsiSetPropertyW(installation, L"NVIDIA_STATUS", L"probe_error");
        MsiSetPropertyW(installation, L"CUDA_READY", L"0");
        return ERROR_SUCCESS;
    }
}
