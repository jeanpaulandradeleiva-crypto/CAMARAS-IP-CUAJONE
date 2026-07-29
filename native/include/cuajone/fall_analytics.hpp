// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include "cuajone/types.hpp"

#include <chrono>
#include <deque>
#include <map>
#include <optional>
#include <span>

namespace cuajone {

bool isValidPosePerson(
    const PoseDetection& pose,
    std::span<const Box> ppe_person_boxes,
    float pose_confidence,
    float iou_threshold);

struct FallConfig {
    std::size_t confirm_frames{12};
    std::size_t reset_frames{20};
    std::chrono::duration<double> alert_cooldown{120.0};
    std::chrono::duration<double> track_ttl{5.0};
    float aspect_ratio{1.05F};
    float torso_angle_degrees{55.0F};
    float descent_ratio{0.12F};
    float near_floor_ratio{0.65F};
};

struct FallResult {
    bool confirmed_now{};
    bool active{};
    float score{};
    float aspect_ratio{};
    float torso_angle{};
    bool recent_descent{};
};

class FallAnalyzer {
public:
    explicit FallAnalyzer(FallConfig config = {});

    FallResult update(
        int track_id,
        const Box& box,
        std::span<const Keypoint> keypoints,
        int frame_height,
        std::chrono::steady_clock::time_point now);
    void prune(std::chrono::steady_clock::time_point now);

private:
    struct State {
        std::deque<std::pair<std::chrono::steady_clock::time_point, float>> centers;
        std::size_t candidate_frames{};
        std::size_t upright_frames{};
        bool active{};
        bool has_alerted{};
        std::chrono::steady_clock::time_point last_alert{};
        std::chrono::steady_clock::time_point recent_descent_until{};
        std::chrono::steady_clock::time_point last_seen{};
    };

    FallConfig config_;
    std::map<int, State> states_;
};

}  // namespace cuajone
