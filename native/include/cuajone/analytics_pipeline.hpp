// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include "cuajone/contracts.hpp"
#include "cuajone/byte_tracker.hpp"
#include "cuajone/fall_analytics.hpp"
#include "cuajone/ppe_analytics.hpp"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace cuajone {

struct AnalyticsPipelineConfig {
    AnalyticsMode mode{AnalyticsMode::PpeFall};
    ByteTrackConfig tracker;
    PpeConfig ppe;
    FallConfig fall;
    float pose_confidence{0.35F};
    float nms_iou{0.45F};
};

struct ObservationFrame {
    std::string contract_version{std::string(kContractVersion)};
    std::string source_id;
    std::uint64_t frame_id{};
    std::int64_t monotonic_timestamp_ms{};
    std::string observed_at;
    int frame_width{};
    int frame_height{};
    std::vector<Detection> ppe_detections;
    std::vector<PoseDetection> pose_detections;
    PpeClassMap ppe_classes;
};

struct ProcessedFrame {
    CanonicalFrameResult canonical;
    std::map<int, PpeAssociation> associations;
};

class AnalyticsPipeline {
public:
    explicit AnalyticsPipeline(AnalyticsPipelineConfig config = {});

    ProcessedFrame process(const ObservationFrame& frame);
    void reset() noexcept;
    [[nodiscard]] std::string_view contractVersion() const noexcept;
    [[nodiscard]] std::string_view runtimeVersion() const noexcept;

private:
    AnalyticsPipelineConfig config_;
    ByteTracker tracker_;
    PpeAnalyzer ppe_analyzer_;
    FallAnalyzer fall_analyzer_;
    std::optional<std::uint64_t> last_frame_id_;
    std::int64_t last_monotonic_timestamp_ms_{};
};

}  // namespace cuajone
