// SPDX-License-Identifier: AGPL-3.0-only

#include "cuajone/byte_tracker.hpp"

#include <BYTETracker.h>
#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace cuajone {
namespace {

bool validBox(const Box& box) noexcept {
    return std::isfinite(box.x1) && std::isfinite(box.y1)
        && std::isfinite(box.x2) && std::isfinite(box.y2)
        && box.x2 > box.x1 && box.y2 > box.y1;
}

}  // namespace

struct ByteTracker::Impl {
    explicit Impl(ByteTrackConfig input)
        : config(input), tracker(
              input.high_confidence_threshold - 0.10F,
              static_cast<int>(input.maximum_age), input.match_threshold,
              input.frame_rate, input.maximum_tracks) {}

    ByteTrackConfig config;
    BYTETracker tracker;
};

ByteTracker::ByteTracker(ByteTrackConfig config) {
    if (!std::isfinite(config.low_confidence_threshold)
        || !std::isfinite(config.high_confidence_threshold)
        || !std::isfinite(config.match_threshold)
        || std::abs(config.low_confidence_threshold - 0.10F) > 0.00001F
        || config.low_confidence_threshold >= config.high_confidence_threshold
        || config.high_confidence_threshold <= 0.20F
        || config.high_confidence_threshold > 1.0F
        || config.match_threshold < 0.0F || config.match_threshold > 1.0F
        || config.maximum_age == 0 || config.maximum_tracks == 0
        || config.maximum_age > static_cast<std::size_t>(std::numeric_limits<int>::max())
        || config.frame_rate <= 0) {
        throw std::invalid_argument("ByteTrack configuration is outside supported ranges");
    }
    impl_ = std::make_unique<Impl>(config);
}

ByteTracker::~ByteTracker() = default;
ByteTracker::ByteTracker(ByteTracker&&) noexcept = default;
ByteTracker& ByteTracker::operator=(ByteTracker&&) noexcept = default;

std::vector<int> ByteTracker::update(std::span<const TrackingDetection> detections) {
    Eigen::MatrixXf rows(static_cast<Eigen::Index>(detections.size()), 5);
    for (std::size_t index = 0; index < detections.size(); ++index) {
        const auto& detection = detections[index];
        if (!validBox(detection.box) || !std::isfinite(detection.confidence)
            || detection.confidence < 0.0F || detection.confidence > 1.0F) {
            throw std::invalid_argument("ByteTrack detections require finite positive boxes and scores in [0, 1]");
        }
        rows(static_cast<Eigen::Index>(index), 0) = detection.box.x1;
        rows(static_cast<Eigen::Index>(index), 1) = detection.box.y1;
        rows(static_cast<Eigen::Index>(index), 2) = detection.box.x2 - detection.box.x1;
        rows(static_cast<Eigen::Index>(index), 3) = detection.box.y2 - detection.box.y1;
        rows(static_cast<Eigen::Index>(index), 4) = detection.confidence;
    }

    const auto tracks = impl_->tracker.process_frame_detections(rows);
    struct Candidate {
        float iou;
        int track_id;
        std::size_t track_index;
        std::size_t detection_index;
    };
    std::vector<Candidate> candidates;
    for (std::size_t track_index = 0; track_index < tracks.size(); ++track_index) {
        const auto box = tracks[track_index].tlbr();
        const Box tracked_box{
            static_cast<float>(box[0]), static_cast<float>(box[1]),
            static_cast<float>(box[2]), static_cast<float>(box[3]),
        };
        for (std::size_t detection_index = 0; detection_index < detections.size(); ++detection_index) {
            const float overlap = intersectionOverUnion(tracked_box, detections[detection_index].box);
            if (overlap > 0.0F) {
                candidates.push_back({
                    overlap, tracks[track_index].get_track_id(), track_index, detection_index,
                });
            }
        }
    }
    std::sort(candidates.begin(), candidates.end(), [](const Candidate& lhs, const Candidate& rhs) {
        return std::tuple{-lhs.iou, lhs.track_id, lhs.detection_index}
            < std::tuple{-rhs.iou, rhs.track_id, rhs.detection_index};
    });

    std::vector<int> ids(detections.size(), -1);
    std::vector<bool> used_tracks(tracks.size(), false);
    for (const auto& candidate : candidates) {
        if (used_tracks[candidate.track_index] || ids[candidate.detection_index] >= 0) continue;
        used_tracks[candidate.track_index] = true;
        ids[candidate.detection_index] = candidate.track_id;
    }
    return ids;
}

std::size_t ByteTracker::activeTrackCount() const noexcept {
    return impl_->tracker.retained_track_count();
}

void ByteTracker::reset() noexcept {
    impl_->tracker.reset();
}

}  // namespace cuajone
