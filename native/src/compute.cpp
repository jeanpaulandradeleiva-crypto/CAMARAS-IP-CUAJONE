// SPDX-License-Identifier: AGPL-3.0-only

#include "cuajone/compute.hpp"

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <dxgi.h>
#endif

namespace cuajone {
namespace {

std::string jsonEscape(std::string_view value) {
    std::ostringstream output;
    for (const unsigned char character : value) {
        switch (character) {
            case '\"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\b': output << "\\b"; break;
            case '\f': output << "\\f"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (character < 0x20U) {
                    output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                           << static_cast<int>(character) << std::dec;
                } else {
                    output << character;
                }
        }
    }
    return output.str();
}

#ifdef _WIN32
std::string utf8(const wchar_t* value) {
    if (value == nullptr || *value == L'\0') return {};
    const int required = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, value, -1, nullptr, 0, nullptr, nullptr);
    if (required <= 1) return {};
    std::string result(static_cast<std::size_t>(required), '\0');
    if (WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, value, -1,
            result.data(), required, nullptr, nullptr) == 0) {
        return {};
    }
    result.resize(static_cast<std::size_t>(required - 1));
    return result;
}

template <typename Function>
Function cudaFunction(HMODULE module, const char* name) {
    return reinterpret_cast<Function>(GetProcAddress(module, name));
}
#endif

}  // namespace

ComputeBackend parseComputeBackend(std::string_view value) {
    if (value == "auto") return ComputeBackend::Auto;
    if (value == "cuda") return ComputeBackend::Cuda;
    if (value == "cpu") return ComputeBackend::Cpu;
    throw std::invalid_argument("--compute must be auto, cuda, or cpu");
}

std::string_view computeBackendName(ComputeBackend backend) noexcept {
    switch (backend) {
        case ComputeBackend::Auto: return "auto";
        case ComputeBackend::Cuda: return "cuda";
        case ComputeBackend::Cpu: return "cpu";
    }
    return "unknown";
}

std::string_view hardwareProbeStatusName(HardwareProbeStatus status) noexcept {
    switch (status) {
        case HardwareProbeStatus::NoNvidiaAdapter: return "no_nvidia_adapter";
        case HardwareProbeStatus::DriverUnavailable: return "driver_unavailable";
        case HardwareProbeStatus::DriverTooOld: return "driver_too_old";
        case HardwareProbeStatus::CudaReady: return "cuda_ready";
        case HardwareProbeStatus::ProbeError: return "probe_error";
    }
    return "probe_error";
}

HardwareProbeResult probeHardware() {
    HardwareProbeResult result;
#ifdef _WIN32
    result.driver_was_loaded = GetModuleHandleW(L"nvcuda.dll") != nullptr;
    IDXGIFactory1* factory = nullptr;
    const HRESULT factory_result = CreateDXGIFactory1(IID_PPV_ARGS(&factory));
    if (FAILED(factory_result) || factory == nullptr) {
        result.status = HardwareProbeStatus::ProbeError;
        result.detail = "CreateDXGIFactory1 failed";
        return result;
    }
    for (UINT index = 0;; ++index) {
        IDXGIAdapter1* adapter = nullptr;
        const HRESULT enumerated = factory->EnumAdapters1(index, &adapter);
        if (enumerated == DXGI_ERROR_NOT_FOUND) break;
        if (FAILED(enumerated) || adapter == nullptr) {
            factory->Release();
            result.status = HardwareProbeStatus::ProbeError;
            result.detail = "DXGI adapter enumeration failed";
            return result;
        }
        DXGI_ADAPTER_DESC1 description{};
        const HRESULT described = adapter->GetDesc1(&description);
        adapter->Release();
        if (FAILED(described)) {
            factory->Release();
            result.status = HardwareProbeStatus::ProbeError;
            result.detail = "DXGI adapter description failed";
            return result;
        }
        if (description.VendorId == 0x10DEU && (description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) == 0) {
            result.adapters.push_back({
                utf8(description.Description),
                description.DeviceId,
                static_cast<std::uint64_t>(description.DedicatedVideoMemory),
            });
        }
    }
    factory->Release();
    if (result.adapters.empty()) {
        result.status = HardwareProbeStatus::NoNvidiaAdapter;
        result.detail = "DXGI found no NVIDIA display adapter";
        return result;
    }

    HMODULE cuda = LoadLibraryExW(L"nvcuda.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (cuda == nullptr) {
        result.status = HardwareProbeStatus::DriverUnavailable;
        result.detail = "nvcuda.dll is unavailable in System32";
        return result;
    }
    using CuInit = int(__stdcall*)(unsigned int);
    using CuDriverGetVersion = int(__stdcall*)(int*);
    using CuDeviceGetCount = int(__stdcall*)(int*);
    using CuDeviceGetName = int(__stdcall*)(char*, int, int);
    using CuDeviceComputeCapability = int(__stdcall*)(int*, int*, int);
    const auto cu_init = cudaFunction<CuInit>(cuda, "cuInit");
    const auto cu_driver_get_version = cudaFunction<CuDriverGetVersion>(cuda, "cuDriverGetVersion");
    const auto cu_device_get_count = cudaFunction<CuDeviceGetCount>(cuda, "cuDeviceGetCount");
    const auto cu_device_get_name = cudaFunction<CuDeviceGetName>(cuda, "cuDeviceGetName");
    const auto cu_device_compute_capability = cudaFunction<CuDeviceComputeCapability>(cuda, "cuDeviceComputeCapability");
    if (!cu_init || !cu_driver_get_version || !cu_device_get_count
        || !cu_device_get_name || !cu_device_compute_capability) {
        FreeLibrary(cuda);
        result.status = HardwareProbeStatus::ProbeError;
        result.detail = "nvcuda.dll does not expose the required CUDA Driver API";
        return result;
    }
    if (cu_init(0) != 0) {
        FreeLibrary(cuda);
        result.status = HardwareProbeStatus::DriverUnavailable;
        result.detail = "CUDA Driver API initialization failed";
        return result;
    }
    int driver_version{};
    if (cu_driver_get_version(&driver_version) != 0) {
        FreeLibrary(cuda);
        result.status = HardwareProbeStatus::ProbeError;
        result.detail = "CUDA Driver API version query failed";
        return result;
    }
    result.driver_version = driver_version;
    if (driver_version < kMinimumCudaDriverApiVersion) {
        FreeLibrary(cuda);
        result.status = HardwareProbeStatus::DriverTooOld;
        result.detail = "CUDA Driver API 12.9 or newer is required";
        return result;
    }
    int device_count{};
    if (cu_device_get_count(&device_count) != 0 || device_count <= 0) {
        FreeLibrary(cuda);
        result.status = HardwareProbeStatus::DriverUnavailable;
        result.detail = "CUDA Driver API found no usable device";
        return result;
    }
    bool compatible_device = false;
    for (int device = 0; device < device_count; ++device) {
        char name[256]{};
        int major{};
        int minor{};
        if (cu_device_get_name(name, static_cast<int>(sizeof(name)), device) != 0
            || cu_device_compute_capability(&major, &minor, device) != 0) {
            FreeLibrary(cuda);
            result.status = HardwareProbeStatus::ProbeError;
            result.detail = "CUDA device capability query failed";
            return result;
        }
        result.cuda_devices.push_back({device, name, major, minor});
        compatible_device = compatible_device || isTensorRtCompatibleComputeCapability(major, minor);
    }
    FreeLibrary(cuda);
    if (!compatible_device) {
        result.status = HardwareProbeStatus::DriverUnavailable;
        result.detail = "CUDA initialized, but TensorRT 11 requires compute capability SM 7.5 or newer";
        return result;
    }
    result.status = HardwareProbeStatus::CudaReady;
    result.detail = "NVIDIA adapter and CUDA Driver API are ready";
#else
    result.status = HardwareProbeStatus::ProbeError;
    result.detail = "Hardware probing is supported only on Windows";
#endif
    return result;
}

std::string hardwareProbeJson(const HardwareProbeResult& result) {
    std::ostringstream output;
    output << "{\"schema_version\":2,\"status\":\""
           << hardwareProbeStatusName(result.status) << "\",\"cuda_ready\":"
           << (result.status == HardwareProbeStatus::CudaReady ? "true" : "false")
           << ",\"driver_was_loaded\":" << (result.driver_was_loaded ? "true" : "false")
           << ",\"minimum_driver_version\":" << kMinimumCudaDriverApiVersion
           << ",\"driver_version\":";
    if (result.driver_version) output << *result.driver_version;
    else output << "null";
    output << ",\"adapters\":[";
    for (std::size_t index = 0; index < result.adapters.size(); ++index) {
        if (index > 0) output << ',';
        const auto& adapter = result.adapters[index];
        output << "{\"name\":\"" << jsonEscape(adapter.name)
               << "\",\"device_id\":" << adapter.device_id
               << ",\"dedicated_video_memory\":" << adapter.dedicated_video_memory << '}';
    }
    output << "],\"cuda_devices\":[";
    for (std::size_t index = 0; index < result.cuda_devices.size(); ++index) {
        if (index > 0) output << ',';
        const auto& device = result.cuda_devices[index];
        output << "{\"device_index\":" << device.device_index
               << ",\"name\":\"" << jsonEscape(device.name)
               << "\",\"compute_major\":" << device.compute_major
               << ",\"compute_minor\":" << device.compute_minor << '}';
    }
    output << "],\"detail\":\"" << jsonEscape(result.detail) << "\"}";
    return output.str();
}

int hardwareProbeExitCode(HardwareProbeStatus status) noexcept {
    switch (status) {
        case HardwareProbeStatus::CudaReady: return 0;
        case HardwareProbeStatus::NoNvidiaAdapter: return 10;
        case HardwareProbeStatus::DriverUnavailable: return 11;
        case HardwareProbeStatus::ProbeError: return 12;
        case HardwareProbeStatus::DriverTooOld: return 13;
    }
    return 12;
}

bool isTensorRtCompatibleComputeCapability(int major, int minor) noexcept {
    return major > 7 || (major == 7 && minor >= 5);
}

int selectCompatibleCudaDevice(
    std::span<const CudaDeviceInfo> devices,
    std::optional<int> requested_device) {
    if (requested_device) {
        const auto selected = std::find_if(devices.begin(), devices.end(), [&](const auto& device) {
            return device.device_index == *requested_device;
        });
        if (selected == devices.end()) {
            throw std::runtime_error("Selected CUDA device index is not available");
        }
        if (!isTensorRtCompatibleComputeCapability(selected->compute_major, selected->compute_minor)) {
            throw std::runtime_error("Selected CUDA device does not meet the TensorRT 11 SM 7.5 minimum");
        }
        return selected->device_index;
    }
    const auto selected = std::find_if(devices.begin(), devices.end(), [](const auto& device) {
        return isTensorRtCompatibleComputeCapability(device.compute_major, device.compute_minor);
    });
    if (selected == devices.end()) {
        throw std::runtime_error("No CUDA device meets the TensorRT 11 SM 7.5 minimum");
    }
    return selected->device_index;
}

ComputeSelection selectComputeBackend(ComputeBackend requested, const ComputeAvailability& available) {
    if (requested == ComputeBackend::Cpu) {
        if (!available.cpu_models_available) {
            throw std::runtime_error("CPU mode requires compatible PPE and pose ONNX models");
        }
        return {ComputeBackend::Cpu, "CPU was explicitly requested"};
    }
    if (requested == ComputeBackend::Cuda) {
        if (!available.tensor_rt_compiled) {
            throw std::runtime_error("CUDA mode is unavailable in this CPU-only build");
        }
        if (available.hardware_status != HardwareProbeStatus::CudaReady) {
            throw std::runtime_error(
                "CUDA mode was requested but the NVIDIA adapter/driver probe is not ready: "
                + std::string(hardwareProbeStatusName(available.hardware_status)));
        }
        if (!available.gpu_models_available) {
            throw std::runtime_error("CUDA mode requires compatible PPE and pose TensorRT engines");
        }
        return {ComputeBackend::Cuda, "CUDA was explicitly requested and passed readiness checks"};
    }
    if (available.tensor_rt_compiled
        && available.hardware_status == HardwareProbeStatus::CudaReady
        && available.gpu_models_available) {
        return {ComputeBackend::Cuda, "Auto selected CUDA after hardware and model readiness checks"};
    }
    if (available.cpu_models_available) {
        return {ComputeBackend::Cpu, "Auto selected CPU because CUDA prerequisites were incomplete"};
    }
    throw std::runtime_error("Auto mode found neither a ready TensorRT path nor compatible ONNX models");
}

std::optional<ComputeBackend> installedComputeBackend() {
#ifdef _WIN32
    wchar_t value[16]{};
    DWORD bytes = sizeof(value);
    const LSTATUS status = RegGetValueW(
        HKEY_LOCAL_MACHINE,
        L"SOFTWARE\\Cuajone PPE Monitor",
        L"ComputeMode",
        RRF_RT_REG_SZ | RRF_SUBKEY_WOW6464KEY,
        nullptr,
        value,
        &bytes);
    if (status == ERROR_FILE_NOT_FOUND || status == ERROR_PATH_NOT_FOUND) return std::nullopt;
    if (status != ERROR_SUCCESS) {
        throw std::runtime_error("Could not read installed compute mode from HKLM");
    }
    std::string converted = utf8(value);
    return parseComputeBackend(converted);
#else
    return std::nullopt;
#endif
}

}  // namespace cuajone
