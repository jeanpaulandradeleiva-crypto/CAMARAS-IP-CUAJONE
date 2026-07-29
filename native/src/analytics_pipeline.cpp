// SPDX-License-Identifier: AGPL-3.0-only

#include "cuajone/analytics_pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace cuajone {
namespace {

bool contains(std::span<const int> ids, int class_id) {
    return std::find(ids.begin(), ids.end(), class_id) != ids.end();
}

std::string safeUrnPart(std::string value) {
    for (char& character : value) {
        const bool allowed = (character >= 'a' && character <= 'z')
            || (character >= 'A' && character <= 'Z')
            || (character >= '0' && character <= '9')
            || character == '.' || character == '_' || character == ':' || character == '-';
        if (!allowed) character = '_';
    }
    return value;
}

}  // namespace

AnalyticsPipeline::AnalyticsPipeline(AnalyticsPipelineConfig config)
    : config_(config), tracker_(config.tracker), ppe_analyzer_(config.ppe),
      fall_analyzer_(config.fall) {
    if (!std::isfinite(config_.pose_confidence) || config_.pose_confidence < 0.0F
        || config_.pose_confidence > 1.0F || !std::isfinite(config_.nms_iou)
        || config_.nms_iou < 0.0F || config_.nms_iou > 1.0F) {
        throw std::invalid_argument("Pipeline confidence and IoU must be finite and in [0, 1]");
    }
}

ProcessedFrame AnalyticsPipeline::process(const ObservationFrame& frame) {
    // Contract and metadata checks form the language-neutral Python/C++ boundary.
    validateContractVersion(frame.contract_version);
    CanonicalFrameResult canonical{
        frame.source_id, frame.frame_id, frame.monotonic_timestamp_ms, frame.observed_at,
        frame.frame_width, frame.frame_height, {}, {},
    };
    validateCanonicalMetadata(canonical);
    // Stateful analytics are reproducible only while frame IDs and monotonic time advance.
    if (last_frame_id_ && frame.frame_id <= *last_frame_id_) {
        throw std::invalid_argument("frame_id must increase strictly until reset");
    }
    if (last_frame_id_ && frame.monotonic_timestamp_ms < last_monotonic_timestamp_ms_) {
        throw std::invalid_argument("monotonic_timestamp_ms must not move backwards until reset");
    }
    if (frame.ppe_classes.person_ids.empty() || frame.ppe_classes.helmet_ids.empty()
        || frame.ppe_classes.vest_ids.empty()) {
        throw std::invalid_argument("PPE semantic classes must include person, helmet, and vest IDs");
    }

    std::vector<PoseDetection> candidates;
    std::vector<Box> ppe_person_boxes;
    for (const auto& detection : frame.ppe_detections) {
        if (contains(frame.ppe_classes.person_ids, detection.class_id)) {
            ppe_person_boxes.push_back(detection.box);
            if (config_.mode == AnalyticsMode::PpeOnly) {
                PoseDetection person;
                person.box = detection.box;
                person.confidence = detection.confidence;
                person.class_id = detection.class_id;
                candidates.push_back(std::move(person));
            }
        }
    }
    if (config_.mode == AnalyticsMode::PpeFall) {
        candidates = frame.pose_detections;
    }

    std::vector<Box> boxes;
    boxes.reserve(candidates.size());
    for (const auto& candidate : candidates) boxes.push_back(candidate.box);
    const auto track_ids = tracker_.update(boxes);
    const float keypoint_threshold = std::clamp(config_.pose_confidence, 0.25F, 0.50F);
    std::vector<TrackedPerson> tracked;
    tracked.reserve(candidates.size());
    for (std::size_t index = 0; index < candidates.size(); ++index) {
        if (track_ids[index] < 0) continue;
        // Preserve pre-refactor IDs: track every decoded pose, then reject invalid people.
        if (config_.mode == AnalyticsMode::PpeFall
            && !isValidPosePerson(
                candidates[index], ppe_person_boxes, config_.pose_confidence, config_.nms_iou)) {
            continue;
        }
        const bool evaluable = isBoxPpeEvaluable(
            candidates[index].box, frame.frame_width, frame.frame_height)
            && (config_.mode == AnalyticsMode::PpeOnly
                || arePoseKeypointsPpeEvaluable(candidates[index].keypoints, keypoint_threshold));
        tracked.push_back({
            track_ids[index], candidates[index].box, candidates[index].confidence,
            candidates[index].keypoints, evaluable,
        });
    }
    const auto associations = associatePpe(tracked, frame.ppe_detections, frame.ppe_classes);
    const auto now = std::chrono::steady_clock::time_point(
        std::chrono::milliseconds(frame.monotonic_timestamp_ms));
    std::size_t event_index{};
    for (const auto& person : tracked) {
        const auto& association = associations.at(person.track_id);
        std::string status = person.ppe_evaluable ? "Evaluating PPE" : "PPE not evaluable";
        std::vector<EventCandidate> candidates_for_person;
        if (auto event = ppe_analyzer_.update(
                person.track_id, association, person.ppe_evaluable, now)) {
            status = event->status;
            candidates_for_person.push_back(std::move(*event));
        }
        if (person.ppe_evaluable) {
            if (const auto stable = ppe_analyzer_.currentStatus(person.track_id)) status = *stable;
        }
        bool fall_active{};
        if (config_.mode == AnalyticsMode::PpeFall) {
            const FallResult fall = fall_analyzer_.update(
                person.track_id, person.box, person.keypoints, frame.frame_height, now);
            fall_active = fall.active;
            if (fall.confirmed_now) {
                candidates_for_person.push_back({
                    person.track_id, "POSIBLE_CAIDA",
                    person.ppe_evaluable ? status : "En evaluación", fall.score,
                });
            }
        }
        canonical.people.push_back({
            person.track_id, person.box, person.confidence, person.ppe_evaluable,
            status, fall_active, std::vector<Keypoint>(person.keypoints.begin(), person.keypoints.end()),
        });
        for (const auto& event : candidates_for_person) {
            const std::string id = "evt-" + safeUrnPart(frame.source_id) + "-"
                + std::to_string(frame.frame_id) + "-" + std::to_string(event.track_id)
                + "-" + std::to_string(event_index++);
            canonical.events.push_back({
                id,
                "urn:cuajone:camera:" + safeUrnPart(frame.source_id),
                event.event_type == "INCUMPLIMIENTO_EPP"
                    ? "com.cuajone.safety.ppe.violation.v1"
                    : "com.cuajone.safety.fall.possible.v1",
                frame.observed_at,
                "track/" + std::to_string(event.track_id),
                frame.frame_id,
                frame.monotonic_timestamp_ms,
                event.track_id,
                event.status,
                event.confidence,
            });
        }
    }
    ppe_analyzer_.prune(now);
    fall_analyzer_.prune(now);
    std::sort(canonical.people.begin(), canonical.people.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.track_id < rhs.track_id;
    });
    std::sort(canonical.events.begin(), canonical.events.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.id < rhs.id;
    });
    // Commit sequence state only after the frame has been fully and canonically processed.
    last_frame_id_ = frame.frame_id;
    last_monotonic_timestamp_ms_ = frame.monotonic_timestamp_ms;
    return {std::move(canonical), associations};
}

void AnalyticsPipeline::reset() noexcept {
    tracker_.reset();
    ppe_analyzer_.reset();
    fall_analyzer_.reset();
    last_frame_id_.reset();
    last_monotonic_timestamp_ms_ = 0;
}

std::string_view AnalyticsPipeline::contractVersion() const noexcept {
    return kContractVersion;
}

std::string_view AnalyticsPipeline::runtimeVersion() const noexcept {
    return kRuntimeVersion;
}

}  // namespace cuajone
