// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include "cuajone/fall_analytics.hpp"
#include "cuajone/compute.hpp"
#include "cuajone/inference_settings.hpp"
#include "cuajone/ppe_analytics.hpp"

#include <array>
#include <chrono>
#include <filesystem>
#include <map>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>

namespace cuajone {

struct RuntimeConfig {
    bool help{};
    bool preflight{};
    bool show_window{};
    bool hardware_probe_json{};
    bool compute_explicit{};
    bool allow_nonperson_pose_class{};
    ComputeBackend compute_backend{ComputeBackend::Auto};
    AnalyticsMode analytics_mode{AnalyticsMode::PpeFall};
    std::string source;
    std::string source_label;
    std::filesystem::path ppe_engine;
    std::filesystem::path pose_engine;
    std::filesystem::path ppe_onnx;
    std::filesystem::path pose_onnx;
    std::filesystem::path output;
    std::optional<std::map<int, std::string>> ppe_labels;
    std::size_t pose_class_count{1};
    std::array<int, 2> pose_keypoint_shape{17, 3};
    std::optional<int> device;
    int image_size{kDefaultImageSize};
    float ppe_confidence{0.30F};
    std::array<float, kPpeOutputLabels.size()> ppe_class_confidences{
        0.30F, 0.30F, 0.30F, 0.30F, 0.30F, 0.30F, 0.30F, 0.30F,
    };
    float pose_confidence{0.35F};
    float nms_iou{0.45F};
    float tracker_high_threshold{0.35F};
    float tracker_match_threshold{0.80F};
    std::size_t max_det{300};
    std::size_t tracker_max_age{30};
    std::size_t tracker_max_tracks{128};
    int tracker_frame_rate{30};
    double target_fps{};
    double reconnect_delay_seconds{5.0};
    double maximum_reconnect_delay_seconds{30.0};
    std::chrono::milliseconds capture_open_timeout{20000};
    std::chrono::milliseconds capture_read_timeout{10000};
    RtspTransport rtsp_transport{RtspTransport::Default};
    PpeConfig ppe;
    FallConfig fall;
};

RuntimeConfig parseCommandLine(int argc, char** argv);
void printHelp(std::ostream& output);
std::string redactSource(const std::string& source);
std::string defaultSourceLabel(const std::string& source);
bool isRtspSource(std::string_view source) noexcept;
void validateRtspSource(const std::string& source);

}  // namespace cuajone
