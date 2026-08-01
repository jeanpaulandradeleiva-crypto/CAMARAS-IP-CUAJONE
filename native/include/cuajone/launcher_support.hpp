// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <filesystem>
#include <utility>
#include <string>
#include <string_view>
#include <vector>

namespace cuajone::launcher {

enum class AnalyticsMode {
    PpeOnly,
    PpeFall,
};

enum class ComputeMode {
    Auto,
    Cuda,
    Cpu,
};

struct LauncherSettings {
    std::wstring source;
    std::filesystem::path output;
    AnalyticsMode analytics_mode{AnalyticsMode::PpeFall};
    ComputeMode compute_mode{ComputeMode::Auto};
    std::filesystem::path ppe_engine;
    std::filesystem::path pose_engine;
    std::filesystem::path ppe_onnx;
    std::filesystem::path pose_onnx;
    std::wstring ppe_labels;
    std::wstring source_label;
    std::vector<std::pair<std::wstring, std::wstring>> runtime_options;
    bool show_window{};
};

struct LaunchPlan {
    std::vector<std::wstring> arguments;
    bool has_cuda_candidate{};
    bool has_cpu_candidate{};
};

std::filesystem::path adjacentOnnxManifest(const std::filesystem::path& model);
LaunchPlan buildLaunchPlan(const LauncherSettings& settings, bool preflight);
std::wstring quoteWindowsArgument(std::wstring_view argument);
std::wstring buildWindowsCommandLine(const std::vector<std::wstring>& arguments);
std::string redactRtspCredentials(std::string_view text);
bool isValidSavedCameraProfileName(std::wstring_view name);
std::wstring_view savedCameraCredentialTargetPrefix();
std::wstring savedCameraCredentialTarget(std::wstring_view name);
void validateRtspCameraUrl(std::wstring_view source);

}  // namespace cuajone::launcher
