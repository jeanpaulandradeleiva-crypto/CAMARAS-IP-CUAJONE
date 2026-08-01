// SPDX-License-Identifier: AGPL-3.0-only

#include "cuajone/fall_analytics.hpp"

#include "cuajone/ppe_analytics.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <vector>

namespace cuajone {
namespace {

float keypointThreshold(float pose_confidence) noexcept {
    return std::clamp(pose_confidence, 0.25F, 0.50F);
}

bool groupVisible(
    std::span<const Keypoint> keypoints,
    std::initializer_list<std::size_t> indices,
    float threshold) noexcept {
    return std::any_of(indices.begin(), indices.end(), [&](std::size_t index) {
        return index < keypoints.size() && keypoints[index].confidence >= threshold;
    });
}

std::optional<Point> averageVisible(
    std::span<const Keypoint> keypoints,
    std::initializer_list<std::size_t> indices,
    float threshold = 0.35F) {
    Point sum{};
    std::size_t count{};
    for (const std::size_t index : indices) {
        if (index >= keypoints.size() || keypoints[index].confidence < threshold) continue;
        sum.x += keypoints[index].x;
        sum.y += keypoints[index].y;
        ++count;
    }
    if (count == 0) return std::nullopt;
    return Point{sum.x / static_cast<float>(count), sum.y / static_cast<float>(count)};
}

}  // namespace

bool isValidPosePerson(
    const PoseDetection& pose,
    std::span<const Box> ppe_person_boxes,
    float pose_confidence,
    float iou_threshold) {
    const float threshold = keypointThreshold(pose_confidence);
    const std::size_t visible = static_cast<std::size_t>(std::count_if(
        pose.keypoints.begin(), pose.keypoints.end(), [threshold](const Keypoint& point) {
            return point.confidence >= threshold;
        }));
    if (visible < 5 || !groupVisible(pose.keypoints, {5, 6}, threshold)
        || !groupVisible(pose.keypoints, {11, 12}, threshold)) {
        return false;
    }
    if (pose.confidence >= std::min(0.85F, pose_confidence + 0.25F)) {
        return true;
    }

    const float minimum_iou = std::clamp(iou_threshold / 2.0F, 0.10F, 0.30F);
    for (const Box& candidate : ppe_person_boxes) {
        if (intersectionOverUnion(pose.box, candidate) >= minimum_iou) return true;
        const float center_x = (candidate.x1 + candidate.x2) / 2.0F;
        const float center_y = (candidate.y1 + candidate.y2) / 2.0F;
        if (center_x >= pose.box.x1 && center_x <= pose.box.x2
            && center_y >= pose.box.y1 && center_y <= pose.box.y2) {
            return true;
        }
    }
    return false;
}

FallAnalyzer::FallAnalyzer(FallConfig config) : config_(config) {
    if (config_.confirm_frames == 0 || config_.reset_frames == 0) {
        throw std::invalid_argument("Fall confirmation and reset frame counts must be positive");
    }
    if (!std::isfinite(config_.alert_cooldown.count()) || config_.alert_cooldown.count() < 0.0
        || !std::isfinite(config_.track_ttl.count()) || config_.track_ttl.count() < 0.0) {
        throw std::invalid_argument("Fall cooldown and track TTL must be finite and non-negative");
    }
    if (!std::isfinite(config_.aspect_ratio) || config_.aspect_ratio <= 0.0F
        || !std::isfinite(config_.torso_angle_degrees)
        || config_.torso_angle_degrees < 0.0F || config_.torso_angle_degrees > 90.0F
        || !std::isfinite(config_.descent_ratio) || config_.descent_ratio < 0.0F
        || !std::isfinite(config_.near_floor_ratio)
        || config_.near_floor_ratio < 0.0F || config_.near_floor_ratio > 1.0F) {
        throw std::invalid_argument("Fall geometry thresholds are outside supported finite ranges");
    }
}

FallResult FallAnalyzer::update(
    int track_id,
    const Box& box,
    std::span<const Keypoint> keypoints,
    int frame_height,
    std::chrono::steady_clock::time_point now) {
    auto& state = states_[track_id];
    state.last_seen = now;
    const float width = std::max(1.0F, box.x2 - box.x1);
    const float height = std::max(1.0F, box.y2 - box.y1);
    const float center_y = (box.y1 + box.y2) / 2.0F;
    const float aspect_ratio = width / height;

    state.centers.emplace_back(now, center_y);
    while (!state.centers.empty() && now - state.centers.front().first > std::chrono::seconds(1)) {
        state.centers.pop_front();
    }
    float minimum_previous = std::numeric_limits<float>::infinity();
    for (std::size_t index = 0; index + 1 < state.centers.size(); ++index) {
        minimum_previous = std::min(minimum_previous, state.centers[index].second);
    }
    const float descent = std::isfinite(minimum_previous) ? center_y - minimum_previous : 0.0F;
    if (descent >= config_.descent_ratio * static_cast<float>(frame_height)) {
        state.recent_descent_until = now + std::chrono::milliseconds(1500);
    }

    float torso_angle = 0.0F;
    if (const auto shoulders = averageVisible(keypoints, {5, 6}); shoulders) {
        if (const auto hips = averageVisible(keypoints, {11, 12}); hips) {
            const float dx = hips->x - shoulders->x;
            const float dy = hips->y - shoulders->y;
            torso_angle = std::atan2(std::abs(dx), std::abs(dy) + 1.0e-6F)
                * 180.0F / std::numbers::pi_v<float>;
        }
    }

    const bool horizontal = aspect_ratio >= config_.aspect_ratio
        || torso_angle >= config_.torso_angle_degrees;
    const bool near_floor = box.y2 >= config_.near_floor_ratio * static_cast<float>(frame_height);
    const bool recent_descent = now <= state.recent_descent_until;
    const bool candidate = horizontal && (near_floor || recent_descent);
    if (candidate) {
        ++state.candidate_frames;
        state.upright_frames = 0;
    } else {
        state.candidate_frames = state.candidate_frames > 2 ? state.candidate_frames - 2 : 0;
        if (aspect_ratio < 0.80F && torso_angle < 35.0F) ++state.upright_frames;
        else state.upright_frames = 0;
    }

    const bool cooldown_elapsed = !state.has_alerted || now - state.last_alert >= config_.alert_cooldown;
    const bool confirmed = state.candidate_frames >= config_.confirm_frames
        && !state.active && cooldown_elapsed;
    if (confirmed) {
        state.active = true;
        state.has_alerted = true;
        state.last_alert = now;
    }
    if (state.active && state.upright_frames >= config_.reset_frames) {
        state.active = false;
        state.candidate_frames = 0;
    }

    const float score = std::min(
        1.0F,
        0.40F * static_cast<float>(horizontal)
            + 0.35F * static_cast<float>(recent_descent)
            + 0.25F * static_cast<float>(near_floor));
    return {confirmed, state.active, score, aspect_ratio, torso_angle, recent_descent};
}

void FallAnalyzer::prune(std::chrono::steady_clock::time_point now) {
    std::erase_if(states_, [&](const auto& item) {
        return now - item.second.last_seen > config_.track_ttl;
    });
}

void FallAnalyzer::reset() noexcept {
    states_.clear();
}

}  // namespace cuajone
