// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include "cuajone/types.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace cuajone {

struct IoUTrackerConfig {
    float minimum_iou{0.30F};
    std::size_t maximum_age{30};
    std::size_t maximum_tracks{128};
};

class IoUTracker {
public:
    explicit IoUTracker(IoUTrackerConfig config = {});

    // IDs are aligned with detections. -1 means the bounded tracker was full.
    std::vector<int> update(std::span<const Box> detections);
    [[nodiscard]] std::size_t activeTrackCount() const noexcept;

private:
    struct Track {
        int id{};
        Box box;
        std::size_t missed{};
    };

    IoUTrackerConfig config_;
    std::vector<Track> tracks_;
    int next_id_{1};
};

}  // namespace cuajone
