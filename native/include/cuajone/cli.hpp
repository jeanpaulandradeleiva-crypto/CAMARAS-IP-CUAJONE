// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include "cuajone/fall_analytics.hpp"
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
    bool allow_nonperson_pose_class{};
    AnalyticsMode analytics_mode{AnalyticsMode::PpeFall};
    std::string source;
    std::string source_label;
    std::filesystem::path ppe_engine;
    std::filesystem::path pose_engine;
    std::filesystem::path output;
    std::optional<std::map<int, std::string>> ppe_labels;
    std::size_t pose_class_count{1};
    std::array<int, 2> pose_keypoint_shape{17, 3};
    int device{};
    float ppe_confidence{0.30F};
    float pose_confidence{0.35F};
    float nms_iou{0.45F};
    float tracker_iou{0.30F};
    std::size_t max_det{300};
    std::size_t tracker_max_age{30};
    std::size_t tracker_max_tracks{128};
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
