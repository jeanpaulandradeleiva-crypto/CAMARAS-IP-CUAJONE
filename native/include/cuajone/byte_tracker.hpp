// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include "cuajone/types.hpp"

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

namespace cuajone {

struct ByteTrackConfig {
    float high_confidence_threshold{0.35F};
    float low_confidence_threshold{0.10F};
    float match_threshold{0.80F};
    std::size_t maximum_age{30};
    std::size_t maximum_tracks{128};
    int frame_rate{30};
};

struct TrackingDetection {
    Box box;
    float confidence{};
};

class ByteTracker {
public:
    explicit ByteTracker(ByteTrackConfig config = {});
    ~ByteTracker();
    ByteTracker(ByteTracker&&) noexcept;
    ByteTracker& operator=(ByteTracker&&) noexcept;
    ByteTracker(const ByteTracker&) = delete;
    ByteTracker& operator=(const ByteTracker&) = delete;

    // IDs are aligned with detections. -1 means no active track was assigned.
    std::vector<int> update(std::span<const TrackingDetection> detections);
    [[nodiscard]] std::size_t activeTrackCount() const noexcept;
    void reset() noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace cuajone
