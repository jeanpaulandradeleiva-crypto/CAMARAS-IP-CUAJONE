// SPDX-License-Identifier: AGPL-3.0-only

#include "cuajone/ppe_analytics.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace cuajone {
namespace {

enum class AssociationRegion { HeadFace, Torso, HandsArms, LowerLegFeet };

struct PpeDefinition {
    PpeItem item;
    std::string_view semantic;
    std::string_view label;
    int class_id;
    AssociationRegion region;
};

constexpr std::array<PpeItem, kPpeItemCount> kRequiredItems{
    PpeItem::Gloves,
    PpeItem::SafetyBoots,
    PpeItem::Vest,
    PpeItem::Respirator,
    PpeItem::HearingProtection,
    PpeItem::HardHat,
    PpeItem::EyeProtection,
};

constexpr std::array<PpeDefinition, kPpeItemCount> kDefinitions{{
    {PpeItem::Gloves, "gloves", "Gloves", 0, AssociationRegion::HandsArms},
    {PpeItem::SafetyBoots, "safety_boots", "Safety_boots", 2, AssociationRegion::LowerLegFeet},
    {PpeItem::Vest, "vest", "Vest", 3, AssociationRegion::Torso},
    {PpeItem::Respirator, "respirator", "respirador", 4, AssociationRegion::HeadFace},
    {PpeItem::HearingProtection, "hearing_protection", "tapaorejas", 5, AssociationRegion::HeadFace},
    {PpeItem::HardHat, "hard_hat", "Hard_hat", 6, AssociationRegion::HeadFace},
    {PpeItem::EyeProtection, "eye_protection", "lentes_protectores", 7, AssociationRegion::HeadFace},
}};

constexpr std::array<std::string_view, 8> kExpectedLabels{
    "Gloves", "Person", "Safety_boots", "Vest", "respirador", "tapaorejas",
    "Hard_hat", "lentes_protectores",
};

const PpeDefinition& definition(PpeItem item) {
    const auto found = std::find_if(kDefinitions.begin(), kDefinitions.end(),
        [item](const PpeDefinition& value) { return value.item == item; });
    if (found == kDefinitions.end()) throw std::logic_error("Unknown PPE item");
    return *found;
}

Box regionForPerson(const Box& person, AssociationRegion strategy) noexcept {
    const float width = std::max(1.0F, person.x2 - person.x1);
    const float height = std::max(1.0F, person.y2 - person.y1);
    switch (strategy) {
    case AssociationRegion::HeadFace:
        return {person.x1 - 0.10F * width, person.y1 - 0.15F * height,
                person.x2 + 0.10F * width, person.y1 + 0.38F * height};
    case AssociationRegion::Torso:
        return {person.x1 + 0.02F * width, person.y1 + 0.15F * height,
                person.x2 - 0.02F * width, person.y1 + 0.72F * height};
    case AssociationRegion::HandsArms:
        return {person.x1 - 0.20F * width, person.y1 + 0.18F * height,
                person.x2 + 0.20F * width, person.y1 + 0.78F * height};
    case AssociationRegion::LowerLegFeet:
        return {person.x1 - 0.10F * width, person.y1 + 0.62F * height,
                person.x2 + 0.10F * width, person.y1 + 1.12F * height};
    }
    return person;
}

float intersectionOverItem(const Box& item, const Box& region) noexcept {
    const float x1 = std::max(item.x1, region.x1);
    const float y1 = std::max(item.y1, region.y1);
    const float x2 = std::min(item.x2, region.x2);
    const float y2 = std::min(item.y2, region.y2);
    const float intersection = std::max(0.0F, x2 - x1) * std::max(0.0F, y2 - y1);
    return intersection / std::max(1.0F, boxArea(item));
}

bool groupVisible(std::span<const Keypoint> keypoints,
    std::initializer_list<std::size_t> indices, float threshold) noexcept {
    return std::any_of(indices.begin(), indices.end(), [&](std::size_t index) {
        return index < keypoints.size() && keypoints[index].confidence >= threshold;
    });
}

bool faceItem(PpeItem item) noexcept {
    return item == PpeItem::Respirator || item == PpeItem::EyeProtection;
}

PpeEvaluation evaluate(const std::array<std::deque<float>, kPpeItemCount>& histories,
    const std::array<std::deque<PpeWearState>, kPpeItemCount>& wear_histories,
    const std::array<std::optional<Detection>, kPpeItemCount>& detections,
    const std::array<PpeWearState, kPpeItemCount>& latest_states,
    const std::array<std::string, kPpeItemCount>& latest_reasons,
    const PpeConfig& config) {
    PpeEvaluation result;
    for (const auto& history : histories) result.samples = std::max(result.samples, history.size());
    result.evaluated = result.samples >= config.minimum_samples;
    result.compliant = result.evaluated;
    result.items.reserve(kPpeItemCount);
    for (const PpeItem item : kRequiredItems) {
        const auto index = static_cast<std::size_t>(item);
        const auto& values = histories[index];
        const auto& states = wear_histories[index];
        const auto present_count = static_cast<float>(std::count_if(values.begin(), values.end(),
            [](float confidence) { return confidence > 0.0F; }));
        const float ratio = values.empty() ? 0.0F : present_count / static_cast<float>(values.size());
        const float confidence_sum = std::accumulate(values.begin(), values.end(), 0.0F);
        const float confidence = present_count == 0.0F ? 0.0F : confidence_sum / present_count;
        const auto count = [&](PpeWearState wanted) {
            return static_cast<float>(std::count(states.begin(), states.end(), wanted));
        };
        const float denominator = static_cast<float>(states.size());
        const bool enabled = config.enabled[index];
        PpeWearState wear_state = PpeWearState::NotVerifiable;
        std::string reason;
        if (!enabled) {
            reason = "DISABLED_BY_POLICY";
        } else if (states.size() < config.minimum_samples) {
            wear_state = latest_states[index];
            reason = latest_reasons[index];
        } else if (count(PpeWearState::PresentCorrectly) / denominator >= config.present_ratio) {
            wear_state = PpeWearState::PresentCorrectly;
            reason = "ASSOCIATED_REGION";
        } else if (count(PpeWearState::PresentIncorrectly) / denominator >= config.present_ratio) {
            wear_state = PpeWearState::PresentIncorrectly;
            reason = "SPATIALLY_INCOMPATIBLE_REGION";
        } else {
            wear_state = PpeWearState::Absent;
            reason = "NO_ASSOCIATED_DETECTION";
        }
        const bool present = wear_state == PpeWearState::PresentCorrectly;
        result.items.push_back({item, enabled, present, ratio, confidence, detections[index],
            enabled, wear_state, std::move(reason)});
        result.compliant = result.compliant && (!enabled || present
            || wear_state == PpeWearState::NotVerifiable);
    }
    return result;
}

const PpeItemState& itemState(const PpeEvaluation& value, PpeItem item) {
    const auto found = std::find_if(value.items.begin(), value.items.end(),
        [item](const PpeItemState& state) { return state.item == item; });
    if (found == value.items.end()) throw std::logic_error("PPE evaluation is incomplete");
    return *found;
}

}  // namespace

std::span<const PpeItem> requiredPpeItems() noexcept { return kRequiredItems; }

std::string_view ppeItemSemantic(PpeItem item) noexcept {
    for (const auto& value : kDefinitions) if (value.item == item) return value.semantic;
    return "unknown";
}

std::string_view ppeItemLabel(PpeItem item) noexcept {
    for (const auto& value : kDefinitions) if (value.item == item) return value.label;
    return "Unknown";
}

std::string_view ppeWearStateName(PpeWearState state) noexcept {
    switch (state) {
    case PpeWearState::PresentCorrectly: return "PRESENTE_CORRECTAMENTE";
    case PpeWearState::PresentIncorrectly: return "PRESENTE_INCORRECTAMENTE";
    case PpeWearState::Absent: return "AUSENTE";
    case PpeWearState::NotVerifiable: return "NO_VERIFICABLE";
    }
    return "NO_VERIFICABLE";
}

std::string normalizeLabel(std::string label) {
    std::transform(label.begin(), label.end(), label.begin(), [](unsigned char character) {
        if (character == '-' || std::isspace(character) != 0) return '_';
        return static_cast<char>(std::tolower(character));
    });
    label.erase(std::unique(label.begin(), label.end(), [](char lhs, char rhs) {
        return lhs == '_' && rhs == '_';
    }), label.end());
    while (!label.empty() && label.front() == '_') label.erase(label.begin());
    while (!label.empty() && label.back() == '_') label.pop_back();
    return label;
}

bool isPersonClassLabel(std::string_view label) {
    const std::string normalized = normalizeLabel(std::string(label));
    return normalized == "person" || normalized == "persona";
}

PpeClassMap resolvePpeClasses(const std::map<int, std::string>& names) {
    if (names.size() != kExpectedLabels.size()) {
        throw std::runtime_error("PPE engine must define exactly 8 ordered labels");
    }
    for (std::size_t id = 0; id < kExpectedLabels.size(); ++id) {
        const auto found = names.find(static_cast<int>(id));
        if (found == names.end() || found->second != kExpectedLabels[id]) {
            throw std::runtime_error("PPE label ID " + std::to_string(id) + " must be '"
                + std::string(kExpectedLabels[id]) + "'");
        }
    }
    PpeClassMap classes;
    classes.person_ids.push_back(1);
    for (const auto& value : kDefinitions) {
        if (!classes.item_ids.emplace(value.item, value.class_id).second) {
            throw std::runtime_error("PPE semantic registry contains an ambiguous item");
        }
    }
    if (classes.item_ids.size() != kPpeItemCount) {
        throw std::runtime_error("PPE semantic registry is missing a required item");
    }
    return classes;
}

bool PpeAssociation::present(PpeItem item) const noexcept { return detections.contains(item); }

std::optional<Detection> PpeAssociation::detection(PpeItem item) const {
    const auto found = detections.find(item);
    return found == detections.end() ? std::nullopt : std::optional<Detection>(found->second);
}

std::optional<Detection> PpeAssociation::incompatibleDetection(PpeItem item) const {
    const auto found = incompatible_detections.find(item);
    return found == incompatible_detections.end() ? std::nullopt : std::optional<Detection>(found->second);
}

std::map<int, PpeAssociation> associatePpe(std::span<const TrackedPerson> people,
    std::span<const Detection> detections, const PpeClassMap& classes,
    const std::array<bool, kPpeItemCount>& enabled) {
    std::map<int, PpeAssociation> associations;
    for (const auto& person : people) associations.try_emplace(person.track_id);
    for (const auto& item : detections) {
        const auto semantic = std::find_if(classes.item_ids.begin(), classes.item_ids.end(),
            [&](const auto& value) { return value.second == item.class_id; });
        if (semantic == classes.item_ids.end()) continue;
        if (!enabled[static_cast<std::size_t>(semantic->first)]) continue;
        const auto& strategy = definition(semantic->first).region;
        const float center_x = (item.box.x1 + item.box.x2) / 2.0F;
        const float center_y = (item.box.y1 + item.box.y2) / 2.0F;
        int best_track_id = -1;
        float best_score = 0.0F;
        for (const auto& person : people) {
            const Box region = regionForPerson(person.box, strategy);
            const bool center_inside = center_x >= region.x1 && center_x <= region.x2
                && center_y >= region.y1 && center_y <= region.y2;
            const float score = intersectionOverItem(item.box, region) + (center_inside ? 0.50F : 0.0F);
            if (score > best_score || (score == best_score && person.track_id < best_track_id)) {
                best_score = score;
                best_track_id = person.track_id;
            }
        }
        if (best_track_id >= 0 && best_score >= 0.35F) {
            auto& selected = associations.at(best_track_id).detections[semantic->first];
            if (selected.confidence == 0.0F || item.confidence > selected.confidence) selected = item;
            continue;
        }
        std::vector<const TrackedPerson*> enclosing;
        for (const auto& person : people) {
            if (center_x >= person.box.x1 && center_x <= person.box.x2
                && center_y >= person.box.y1 && center_y <= person.box.y2) {
                enclosing.push_back(&person);
            }
        }
        if (enclosing.size() == 1) {
            auto& selected = associations.at(enclosing.front()->track_id).incompatible_detections[semantic->first];
            if (selected.confidence == 0.0F || item.confidence > selected.confidence) selected = item;
        }
    }
    return associations;
}

bool isBoxPpeEvaluable(const Box& box, int frame_width, int frame_height) noexcept {
    const int margin = std::max(8, static_cast<int>(std::min(frame_width, frame_height) * 0.01F));
    return box.y2 < static_cast<float>(frame_height - margin)
        && (box.y2 - box.y1) / std::max(1.0F, static_cast<float>(frame_height)) >= 0.12F;
}

bool arePoseKeypointsPpeEvaluable(std::span<const Keypoint> keypoints,
    float keypoint_threshold) noexcept {
    return groupVisible(keypoints, {0, 1, 2, 3, 4}, keypoint_threshold)
        && groupVisible(keypoints, {5, 6}, keypoint_threshold)
        && groupVisible(keypoints, {11, 12}, keypoint_threshold);
}

bool hasFrontalFaceEvidence(std::span<const Keypoint> keypoints, float keypoint_threshold) noexcept {
    if (keypoints.size() < 5) return false;
    const bool nose = keypoints[0].confidence >= keypoint_threshold;
    const bool left_eye = keypoints[1].confidence >= keypoint_threshold;
    const bool right_eye = keypoints[2].confidence >= keypoint_threshold;
    const bool left_ear = keypoints[3].confidence >= keypoint_threshold;
    const bool right_ear = keypoints[4].confidence >= keypoint_threshold;
    return nose && left_eye && right_eye && left_ear && right_ear;
}

std::string ppeStatus(const PpeEvaluation& evaluation) {
    if (!evaluation.evaluated) return "Evaluando EPP";
    if (evaluation.compliant) return "EPP Completo";
    std::string status{"Falta: "};
    bool first = true;
    for (const auto& item : evaluation.items) {
        if (!item.enabled || item.wear_state == PpeWearState::PresentCorrectly
            || item.wear_state == PpeWearState::NotVerifiable) continue;
        if (!first) status += ", ";
        status += ppeItemLabel(item.item);
        first = false;
    }
    return first ? "EPP No verificable" : status;
}

std::string legacyPpeStatus(const PpeEvaluation& evaluation) {
    if (!evaluation.evaluated) return "Evaluating PPE";
    const bool helmet = itemState(evaluation, PpeItem::HardHat).present;
    const bool vest = itemState(evaluation, PpeItem::Vest).present;
    if (helmet && vest) return "EPP Completo";
    if (helmet) return "Falta Chaleco";
    if (vest) return "Falta Casco";
    return "Sin Casco y Chaleco";
}

PpeAnalyzer::PpeAnalyzer(PpeConfig config) : config_(config) {
    if (config_.window == 0 || config_.minimum_samples == 0 || config_.minimum_samples > config_.window) {
        throw std::invalid_argument("PPE minimum samples must be in [1, window]");
    }
    if (!std::isfinite(config_.present_ratio) || config_.present_ratio < 0.0F || config_.present_ratio > 1.0F) {
        throw std::invalid_argument("PPE present ratio must be finite and in [0, 1]");
    }
    if (!std::isfinite(config_.alert_cooldown.count()) || config_.alert_cooldown.count() < 0.0
        || !std::isfinite(config_.track_ttl.count()) || config_.track_ttl.count() < 0.0) {
        throw std::invalid_argument("PPE cooldown and track TTL must be finite and non-negative");
    }
}

std::optional<EventCandidate> PpeAnalyzer::update(int track_id, const PpeAssociation& association,
    bool evaluable, std::chrono::steady_clock::time_point now) {
    auto& state = states_[track_id];
    state.last_seen = now;
    for (const PpeItem item : kRequiredItems) {
        const auto index = static_cast<std::size_t>(item);
        auto& history = state.histories[index];
        auto& wear_history = state.wear_histories[index];
        const auto detection = association.detection(item);
        const auto incompatible = association.incompatibleDetection(item);
        PpeWearState wear_state = PpeWearState::NotVerifiable;
        std::string reason = "VISIBILITY_INSUFFICIENT";
        if (!config_.enabled[index]) {
            reason = "DISABLED_BY_POLICY";
        } else if (faceItem(item) && !evaluable) {
            reason = "VISIBILITY_INSUFFICIENT";
        } else if (detection) {
            wear_state = PpeWearState::PresentCorrectly;
            reason = "ASSOCIATED_REGION";
        } else if (incompatible) {
            wear_state = PpeWearState::PresentIncorrectly;
            reason = "SPATIALLY_INCOMPATIBLE_REGION";
        } else {
            wear_state = PpeWearState::Absent;
            reason = "NO_ASSOCIATED_DETECTION";
        }
        state.latest_states[index] = wear_state;
        state.latest_reasons[index] = reason;
        state.detections[index] = detection ? detection : incompatible;
        if (wear_state != PpeWearState::NotVerifiable) {
            history.push_back(detection ? detection->confidence : 0.0F);
            wear_history.push_back(wear_state);
            while (history.size() > config_.window) history.pop_front();
            while (wear_history.size() > config_.window) wear_history.pop_front();
        }
    }
    const PpeEvaluation evaluation = evaluate(
        state.histories, state.wear_histories, state.detections,
        state.latest_states, state.latest_reasons, config_);
    if (!evaluation.evaluated) return std::nullopt;
    const std::string status = ppeStatus(evaluation);
    const bool changed = status != state.last_status;
    const bool cooldown_elapsed = !state.has_alerted || now - state.last_alert >= config_.alert_cooldown;
    state.last_status = status;
    const bool violating = std::ranges::any_of(evaluation.items, [](const PpeItemState& item) {
        return item.enabled && (item.wear_state == PpeWearState::Absent
            || item.wear_state == PpeWearState::PresentIncorrectly);
    });
    if (!violating || (!changed && !cooldown_elapsed)) return std::nullopt;
    state.last_alert = now;
    state.has_alerted = true;
    float confidence{};
    for (const auto& item : evaluation.items) {
        if (item.wear_state == PpeWearState::Absent
            || item.wear_state == PpeWearState::PresentIncorrectly) {
            confidence = std::max(confidence, 1.0F - item.ratio);
        }
    }
    return EventCandidate{track_id, "INCUMPLIMIENTO_EPP", status, confidence, evaluation};
}

std::optional<PpeEvaluation> PpeAnalyzer::currentEvaluation(int track_id) const {
    const auto found = states_.find(track_id);
    if (found == states_.end()) return std::nullopt;
    std::size_t samples{};
    for (const auto& history : found->second.histories) samples = std::max(samples, history.size());
    if (samples == 0) return std::nullopt;
    return evaluate(found->second.histories, found->second.wear_histories,
        found->second.detections, found->second.latest_states,
        found->second.latest_reasons, config_);
}

void PpeAnalyzer::prune(std::chrono::steady_clock::time_point now) {
    std::erase_if(states_, [&](const auto& item) { return now - item.second.last_seen > config_.track_ttl; });
}

void PpeAnalyzer::reset() noexcept { states_.clear(); }

}  // namespace cuajone
