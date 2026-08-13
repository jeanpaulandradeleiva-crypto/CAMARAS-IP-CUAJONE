// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include "cuajone/inference_settings.hpp"
#include "cuajone/types.hpp"

#include <array>
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

enum class UiLanguage {
    English,
    Spanish,
};

enum class ThemeMode {
    Light,
    Dark,
};

struct OperatorPreferences {
    std::size_t schema_version{1};
    UiLanguage language{UiLanguage::English};
    ThemeMode theme{ThemeMode::Light};
    int image_size{kDefaultImageSize};
    std::array<float, kPpeOutputLabels.size()> ppe_class_confidences{
        0.30F, 0.30F, 0.30F, 0.30F, 0.30F, 0.30F, 0.30F, 0.30F,
    };
    std::array<bool, kPpeItemCount> ppe_enabled{true, true, true, true, true, true, true};
    bool show_window{true};
};

struct ManagedModelSet {
    std::filesystem::path root;
    std::filesystem::path ppe_engine;
    std::filesystem::path pose_engine;
    std::filesystem::path ppe_onnx;
    std::filesystem::path pose_onnx;
    bool tensor_rt_complete{};
    bool onnx_complete{};
};

struct LauncherSettings {
    std::wstring source;
    std::filesystem::path output;
    AnalyticsMode analytics_mode{AnalyticsMode::PpeFall};
    ComputeMode compute_mode{ComputeMode::Auto};
    std::filesystem::path managed_model_root;
    std::wstring source_label;
    std::vector<std::pair<std::wstring, std::wstring>> runtime_options;
    int image_size{kDefaultImageSize};
    std::array<float, kPpeOutputLabels.size()> ppe_class_confidences{
        0.30F, 0.30F, 0.30F, 0.30F, 0.30F, 0.30F, 0.30F, 0.30F,
    };
    std::array<bool, kPpeItemCount> ppe_enabled{true, true, true, true, true, true, true};
    bool show_window{true};
};

struct LaunchPlan {
    std::vector<std::wstring> arguments;
    bool has_cuda_candidate{};
    bool has_cpu_candidate{};
};

std::filesystem::path adjacentOnnxManifest(const std::filesystem::path& model);
ManagedModelSet resolveManagedModelSet(
    const std::filesystem::path& root,
    bool pose_required);
LaunchPlan buildLaunchPlan(const LauncherSettings& settings, bool preflight);
float parsePpeConfidenceThreshold(std::wstring_view text);
std::wstring formatPpeConfidenceThreshold(float value);
OperatorPreferences parseOperatorPreferences(std::string_view text);
std::string serializeOperatorPreferences(const OperatorPreferences& preferences);
OperatorPreferences loadOperatorPreferences(const std::filesystem::path& path) noexcept;
void saveOperatorPreferencesAtomic(
    const std::filesystem::path& path,
    const OperatorPreferences& preferences);
std::vector<std::string_view> visibleLauncherControlKeys();
std::wstring quoteWindowsArgument(std::wstring_view argument);
std::wstring buildWindowsCommandLine(const std::vector<std::wstring>& arguments);
std::string redactRtspCredentials(std::string_view text);
bool isValidSavedCameraProfileName(std::wstring_view name);
std::wstring_view savedCameraCredentialTargetPrefix();
std::wstring savedCameraCredentialTarget(std::wstring_view name);
void validateRtspCameraUrl(std::wstring_view source);

}  // namespace cuajone::launcher
