// SPDX-License-Identifier: AGPL-3.0-only

#include "cuajone/ppe_analytics.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>
#include <unordered_set>

namespace cuajone {
namespace {

const std::unordered_set<std::string> kPersonLabels{"person", "persona"};
const std::unordered_set<std::string> kHelmetLabels{
    "hard_hat", "hardhat", "helmet", "safety_helmet", "casco"};
const std::unordered_set<std::string> kVestLabels{
    "vest", "safety_vest", "reflective_vest", "chaleco", "chaleco_reflectivo"};

bool includes(std::span<const int> ids, int class_id) {
    return std::find(ids.begin(), ids.end(), class_id) != ids.end();
}

Box regionForPerson(const Box& person, bool helmet) noexcept {
    const float width = std::max(1.0F, person.x2 - person.x1);
    const float height = std::max(1.0F, person.y2 - person.y1);
    if (helmet) {
        return {
            person.x1 - 0.10F * width,
            person.y1 - 0.15F * height,
            person.x2 + 0.10F * width,
            person.y1 + 0.38F * height,
        };
    }
    return {
        person.x1 + 0.02F * width,
        person.y1 + 0.15F * height,
        person.x2 - 0.02F * width,
        person.y1 + 0.78F * height,
    };
}

float intersectionOverItem(const Box& item, const Box& region) noexcept {
    const float x1 = std::max(item.x1, region.x1);
    const float y1 = std::max(item.y1, region.y1);
    const float x2 = std::min(item.x2, region.x2);
    const float y2 = std::min(item.y2, region.y2);
    const float intersection = std::max(0.0F, x2 - x1) * std::max(0.0F, y2 - y1);
    return intersection / std::max(1.0F, boxArea(item));
}

bool groupVisible(
    std::span<const Keypoint> keypoints,
    std::initializer_list<std::size_t> indices,
    float threshold) noexcept {
    return std::any_of(indices.begin(), indices.end(), [&](std::size_t index) {
        return index < keypoints.size() && keypoints[index].confidence >= threshold;
    });
}

}  // namespace

std::string normalizeLabel(std::string label) {
    std::transform(label.begin(), label.end(), label.begin(), [](unsigned char character) {
        if (character == '-' || std::isspace(character) != 0) return '_';
        return static_cast<char>(std::tolower(character));
    });
    label.erase(std::unique(label.begin(), label.end(), [](char lhs, char rhs) {
        return lhs == '_' && rhs == '_';
    }), label.end());
    while (!label.empty() && (label.front() == '_' || std::isspace(static_cast<unsigned char>(label.front())) != 0)) {
        label.erase(label.begin());
    }
    while (!label.empty() && (label.back() == '_' || std::isspace(static_cast<unsigned char>(label.back())) != 0)) {
        label.pop_back();
    }
    return label;
}

bool isPersonClassLabel(std::string_view label) {
    return kPersonLabels.contains(normalizeLabel(std::string(label)));
}

PpeClassMap resolvePpeClasses(const std::map<int, std::string>& names) {
    PpeClassMap classes;
    for (const auto& [id, raw_name] : names) {
        const std::string name = normalizeLabel(raw_name);
        if (isPersonClassLabel(name)) classes.person_ids.push_back(id);
        if (kHelmetLabels.contains(name)) classes.helmet_ids.push_back(id);
        if (kVestLabels.contains(name)) classes.vest_ids.push_back(id);
    }
    if (classes.person_ids.empty() || classes.helmet_ids.empty() || classes.vest_ids.empty()) {
        throw std::runtime_error(
            "PPE engine labels must include recognized Person, helmet, and vest classes");
    }
    return classes;
}

std::map<int, PpeAssociation> associatePpe(
    std::span<const TrackedPerson> people,
    std::span<const Detection> detections,
    const PpeClassMap& classes) {
    std::map<int, PpeAssociation> associations;
    for (const auto& person : people) {
        associations.try_emplace(person.track_id);
    }

    for (const auto& item : detections) {
        const bool helmet = includes(classes.helmet_ids, item.class_id);
        const bool vest = includes(classes.vest_ids, item.class_id);
        if (!helmet && !vest) {
            continue;
        }
        const float center_x = (item.box.x1 + item.box.x2) / 2.0F;
        const float center_y = (item.box.y1 + item.box.y2) / 2.0F;
        int best_track_id = -1;
        float best_score = 0.0F;
        for (const auto& person : people) {
            if (!person.ppe_evaluable) {
                continue;
            }
            const Box region = regionForPerson(person.box, helmet);
            const bool center_inside = center_x >= region.x1 && center_x <= region.x2
                && center_y >= region.y1 && center_y <= region.y2;
            const float score = intersectionOverItem(item.box, region) + (center_inside ? 0.50F : 0.0F);
            if (score > best_score || (score == best_score && person.track_id < best_track_id)) {
                best_score = score;
                best_track_id = person.track_id;
            }
        }
        if (best_track_id < 0 || best_score < 0.35F) {
            continue;
        }
        auto& association = associations.at(best_track_id);
        auto& selected = helmet ? association.helmet_detection : association.vest_detection;
        if (!selected || item.confidence > selected->confidence) {
            selected = item;
            if (helmet) association.helmet = true;
            else association.vest = true;
        }
    }
    return associations;
}

bool isBoxPpeEvaluable(const Box& box, int frame_width, int frame_height) noexcept {
    const int margin = std::max(8, static_cast<int>(std::min(frame_width, frame_height) * 0.01F));
    if (box.y2 >= static_cast<float>(frame_height - margin)) {
        return false;
    }
    return (box.y2 - box.y1) / std::max(1.0F, static_cast<float>(frame_height)) >= 0.12F;
}

bool arePoseKeypointsPpeEvaluable(
    std::span<const Keypoint> keypoints,
    float keypoint_threshold) noexcept {
    return groupVisible(keypoints, {0, 1, 2, 3, 4}, keypoint_threshold)
        && groupVisible(keypoints, {5, 6}, keypoint_threshold)
        && groupVisible(keypoints, {11, 12}, keypoint_threshold);
}

PpeAnalyzer::PpeAnalyzer(PpeConfig config) : config_(config) {
    if (config_.window == 0 || config_.minimum_samples == 0
        || config_.minimum_samples > config_.window) {
        throw std::invalid_argument("PPE minimum samples must be in [1, window]");
    }
    if (!std::isfinite(config_.present_ratio) || config_.present_ratio < 0.0F
        || config_.present_ratio > 1.0F) {
        throw std::invalid_argument("PPE present ratio must be finite and in [0, 1]");
    }
    if (!std::isfinite(config_.alert_cooldown.count()) || config_.alert_cooldown.count() < 0.0
        || !std::isfinite(config_.track_ttl.count()) || config_.track_ttl.count() < 0.0) {
        throw std::invalid_argument("PPE cooldown and track TTL must be finite and non-negative");
    }
}

std::optional<EventCandidate> PpeAnalyzer::update(
    int track_id,
    const PpeAssociation& association,
    bool evaluable,
    std::chrono::steady_clock::time_point now) {
    auto& state = states_[track_id];
    state.last_seen = now;
    if (!evaluable) {
        return std::nullopt;
    }
    state.helmet_history.push_back(association.helmet);
    state.vest_history.push_back(association.vest);
    while (state.helmet_history.size() > config_.window) state.helmet_history.pop_front();
    while (state.vest_history.size() > config_.window) state.vest_history.pop_front();
    const std::size_t samples = std::min(state.helmet_history.size(), state.vest_history.size());
    if (samples < config_.minimum_samples) {
        return std::nullopt;
    }

    const auto trueCount = [](const std::deque<bool>& values) {
        return static_cast<float>(std::count(values.begin(), values.end(), true));
    };
    const float helmet_ratio = trueCount(state.helmet_history) / static_cast<float>(state.helmet_history.size());
    const float vest_ratio = trueCount(state.vest_history) / static_cast<float>(state.vest_history.size());
    const bool helmet = helmet_ratio >= config_.present_ratio;
    const bool vest = vest_ratio >= config_.present_ratio;
    std::string status;
    if (helmet && vest) status = "EPP Completo";
    else if (helmet) status = "Falta Chaleco";
    else if (vest) status = "Falta Casco";
    else status = "Sin Casco y Chaleco";

    const bool violation = !(helmet && vest);
    const bool changed = status != state.last_status;
    const bool cooldown_elapsed = !state.has_alerted || now - state.last_alert >= config_.alert_cooldown;
    state.last_status = status;
    if (!violation || (!changed && !cooldown_elapsed)) {
        return std::nullopt;
    }

    state.last_alert = now;
    state.has_alerted = true;
    return EventCandidate{
        track_id,
        "INCUMPLIMIENTO_EPP",
        std::move(status),
        std::min(1.0F, std::max(1.0F - helmet_ratio, 1.0F - vest_ratio)),
    };
}

std::optional<std::string> PpeAnalyzer::currentStatus(int track_id) const {
    const auto iterator = states_.find(track_id);
    if (iterator == states_.end() || iterator->second.last_status.empty()) return std::nullopt;
    return iterator->second.last_status;
}

void PpeAnalyzer::prune(std::chrono::steady_clock::time_point now) {
    std::erase_if(states_, [&](const auto& item) {
        return now - item.second.last_seen > config_.track_ttl;
    });
}

void PpeAnalyzer::reset() noexcept {
    states_.clear();
}

}  // namespace cuajone
