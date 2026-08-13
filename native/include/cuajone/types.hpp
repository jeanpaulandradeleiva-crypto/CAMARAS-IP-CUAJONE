// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace cuajone {

enum class RtspTransport {
    Default,
    Tcp,
    Udp,
};

enum class AnalyticsMode {
    PpeOnly,
    PpeFall,
};

struct Point {
    float x{};
    float y{};
};

struct Box {
    float x1{};
    float y1{};
    float x2{};
    float y2{};
};

struct Keypoint {
    float x{};
    float y{};
    float confidence{};
};

struct Detection {
    Box box;
    float confidence{};
    int class_id{};
};

struct PoseDetection : Detection {
    std::vector<Keypoint> keypoints;
};

enum class PpeItem : std::size_t {
    Gloves,
    SafetyBoots,
    Vest,
    Respirator,
    HearingProtection,
    HardHat,
    EyeProtection,
    Count,
};

inline constexpr std::size_t kPpeItemCount = static_cast<std::size_t>(PpeItem::Count);

enum class PpeWearState {
    PresentCorrectly,
    PresentIncorrectly,
    Absent,
    NotVerifiable,
};

struct PpeItemState {
    PpeItem item{};
    bool required{true};
    bool present{};
    float ratio{};
    float confidence{};
    std::optional<Detection> detection;
    bool enabled{true};
    PpeWearState wear_state{PpeWearState::NotVerifiable};
    std::string reason;
};

struct PpeEvaluation {
    bool evaluated{};
    bool compliant{};
    std::size_t samples{};
    std::vector<PpeItemState> items;
};

struct EventCandidate {
    int track_id{};
    std::string event_type;
    std::string status;
    float confidence{};
    std::optional<PpeEvaluation> ppe;
};

float boxArea(const Box& box) noexcept;
float intersectionOverUnion(const Box& lhs, const Box& rhs) noexcept;

}  // namespace cuajone
