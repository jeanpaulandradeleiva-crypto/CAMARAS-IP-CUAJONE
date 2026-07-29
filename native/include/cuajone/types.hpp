// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <cstddef>
#include <cstdint>
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

struct EventCandidate {
    int track_id{};
    std::string event_type;
    std::string status;
    float confidence{};
};

float boxArea(const Box& box) noexcept;
float intersectionOverUnion(const Box& lhs, const Box& rhs) noexcept;

}  // namespace cuajone
