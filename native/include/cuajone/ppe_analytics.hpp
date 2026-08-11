// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include "cuajone/types.hpp"

#include <array>
#include <chrono>
#include <deque>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace cuajone {

struct PpeClassMap {
    std::vector<int> person_ids;
    std::map<PpeItem, int> item_ids;
};

PpeClassMap resolvePpeClasses(const std::map<int, std::string>& names);
std::string normalizeLabel(std::string label);
bool isPersonClassLabel(std::string_view label);
std::span<const PpeItem> requiredPpeItems() noexcept;
std::string_view ppeItemSemantic(PpeItem item) noexcept;
std::string_view ppeItemLabel(PpeItem item) noexcept;
std::string ppeStatus(const PpeEvaluation& evaluation);
std::string legacyPpeStatus(const PpeEvaluation& evaluation);

struct TrackedPerson {
    int track_id{};
    Box box;
    float confidence{};
    std::span<const Keypoint> keypoints;
    bool ppe_evaluable{};
};

struct PpeAssociation {
    std::map<PpeItem, Detection> detections;

    [[nodiscard]] bool present(PpeItem item) const noexcept;
    [[nodiscard]] std::optional<Detection> detection(PpeItem item) const;
};

std::map<int, PpeAssociation> associatePpe(
    std::span<const TrackedPerson> people,
    std::span<const Detection> detections,
    const PpeClassMap& classes);

bool isBoxPpeEvaluable(const Box& box, int frame_width, int frame_height) noexcept;
bool arePoseKeypointsPpeEvaluable(
    std::span<const Keypoint> keypoints,
    float keypoint_threshold) noexcept;

struct PpeConfig {
    std::size_t window{20};
    std::size_t minimum_samples{12};
    float present_ratio{0.35F};
    std::chrono::duration<double> alert_cooldown{60.0};
    std::chrono::duration<double> track_ttl{5.0};
};

class PpeAnalyzer {
public:
    explicit PpeAnalyzer(PpeConfig config = {});

    std::optional<EventCandidate> update(
        int track_id,
        const PpeAssociation& association,
        bool evaluable,
        std::chrono::steady_clock::time_point now);
    [[nodiscard]] std::optional<PpeEvaluation> currentEvaluation(int track_id) const;
    void prune(std::chrono::steady_clock::time_point now);
    void reset() noexcept;

private:
    struct State {
        std::array<std::deque<float>, kPpeItemCount> histories;
        std::array<std::optional<Detection>, kPpeItemCount> detections;
        std::string last_status;
        std::chrono::steady_clock::time_point last_seen{};
        std::chrono::steady_clock::time_point last_alert{};
        bool has_alerted{};
    };

    PpeConfig config_;
    std::map<int, State> states_;
};

}  // namespace cuajone
