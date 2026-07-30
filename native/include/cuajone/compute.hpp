// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace cuajone {

enum class ComputeBackend {
    Auto,
    Cuda,
    Cpu,
};

enum class HardwareProbeStatus {
    NoNvidiaAdapter,
    DriverUnavailable,
    DriverTooOld,
    CudaReady,
    ProbeError,
};

struct NvidiaAdapterInfo {
    std::string name;
    std::uint32_t device_id{};
    std::uint64_t dedicated_video_memory{};
};

struct CudaDeviceInfo {
    int device_index{};
    std::string name;
    int compute_major{};
    int compute_minor{};
};

inline constexpr int kMinimumCudaDriverApiVersion = 12090;

struct HardwareProbeResult {
    HardwareProbeStatus status{HardwareProbeStatus::ProbeError};
    bool driver_was_loaded{};
    std::vector<NvidiaAdapterInfo> adapters;
    std::vector<CudaDeviceInfo> cuda_devices;
    std::optional<int> driver_version;
    std::string detail;
};

struct ComputeAvailability {
    HardwareProbeStatus hardware_status{HardwareProbeStatus::ProbeError};
    bool tensor_rt_compiled{};
    bool gpu_models_available{};
    bool cpu_models_available{};
};

struct ComputeSelection {
    ComputeBackend backend{ComputeBackend::Cpu};
    std::string reason;
};

ComputeBackend parseComputeBackend(std::string_view value);
std::string_view computeBackendName(ComputeBackend backend) noexcept;
std::string_view hardwareProbeStatusName(HardwareProbeStatus status) noexcept;
HardwareProbeResult probeHardware();
std::string hardwareProbeJson(const HardwareProbeResult& result);
int hardwareProbeExitCode(HardwareProbeStatus status) noexcept;
bool isTensorRtCompatibleComputeCapability(int major, int minor) noexcept;
int selectCompatibleCudaDevice(
    std::span<const CudaDeviceInfo> devices,
    std::optional<int> requested_device);
ComputeSelection selectComputeBackend(ComputeBackend requested, const ComputeAvailability& available);
std::optional<ComputeBackend> installedComputeBackend();

}  // namespace cuajone
