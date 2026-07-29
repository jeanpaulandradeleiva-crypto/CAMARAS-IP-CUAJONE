// SPDX-License-Identifier: AGPL-3.0-only

#include "cuajone/iou_tracker.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <tuple>

namespace cuajone {

IoUTracker::IoUTracker(IoUTrackerConfig config) : config_(config) {
    if (!std::isfinite(config_.minimum_iou) || config_.minimum_iou < 0.0F
        || config_.minimum_iou > 1.0F) {
        throw std::invalid_argument("Tracker minimum IoU must be finite and in [0, 1]");
    }
    if (config_.maximum_age == 0 || config_.maximum_tracks == 0) {
        throw std::invalid_argument("Tracker maximum age and track count must be positive");
    }
}

std::vector<int> IoUTracker::update(std::span<const Box> detections) {
    struct Candidate {
        float iou;
        std::size_t track_index;
        std::size_t detection_index;
        int track_id;
    };

    for (auto& track : tracks_) {
        ++track.missed;
    }

    std::vector<Candidate> candidates;
    for (std::size_t track_index = 0; track_index < tracks_.size(); ++track_index) {
        for (std::size_t detection_index = 0; detection_index < detections.size(); ++detection_index) {
            const float iou = intersectionOverUnion(tracks_[track_index].box, detections[detection_index]);
            if (iou >= config_.minimum_iou) {
                candidates.push_back({iou, track_index, detection_index, tracks_[track_index].id});
            }
        }
    }
    std::sort(candidates.begin(), candidates.end(), [](const Candidate& lhs, const Candidate& rhs) {
        return std::tuple{-lhs.iou, lhs.track_id, lhs.detection_index}
            < std::tuple{-rhs.iou, rhs.track_id, rhs.detection_index};
    });

    std::vector<bool> used_tracks(tracks_.size(), false);
    std::vector<int> ids(detections.size(), -1);
    for (const auto& candidate : candidates) {
        if (used_tracks[candidate.track_index] || ids[candidate.detection_index] != -1) {
            continue;
        }
        auto& track = tracks_[candidate.track_index];
        track.box = detections[candidate.detection_index];
        track.missed = 0;
        used_tracks[candidate.track_index] = true;
        ids[candidate.detection_index] = track.id;
    }

    tracks_.erase(
        std::remove_if(tracks_.begin(), tracks_.end(), [this](const Track& track) {
            return track.missed > config_.maximum_age;
        }),
        tracks_.end());

    for (std::size_t detection_index = 0; detection_index < detections.size(); ++detection_index) {
        if (ids[detection_index] != -1 || tracks_.size() >= config_.maximum_tracks) {
            continue;
        }
        const int id = next_id_++;
        tracks_.push_back({id, detections[detection_index], 0});
        ids[detection_index] = id;
    }
    std::sort(tracks_.begin(), tracks_.end(), [](const Track& lhs, const Track& rhs) {
        return lhs.id < rhs.id;
    });
    return ids;
}

std::size_t IoUTracker::activeTrackCount() const noexcept {
    return tracks_.size();
}

}  // namespace cuajone
