// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include "cuajone/types.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace cuajone {

inline constexpr std::string_view kContractVersion{"1.0.0"};
inline constexpr std::string_view kRuntimeVersion{"0.1.0-internal.4-dev"};
inline constexpr int kCanonicalDecimalDigits{6};

struct CanonicalPerson {
    int track_id{};
    Box box;
    float confidence{};
    bool ppe_evaluable{};
    std::string ppe_status;
    bool fall_active{};
    std::vector<Keypoint> keypoints;
};

struct CanonicalEvent {
    std::string id;
    std::string source;
    std::string type;
    std::string time;
    std::string subject;
    std::uint64_t frame_id{};
    std::int64_t monotonic_timestamp_ms{};
    int track_id{};
    std::string status;
    float confidence{};
};

struct CanonicalFrameResult {
    std::string source_id;
    std::uint64_t frame_id{};
    std::int64_t monotonic_timestamp_ms{};
    std::string observed_at;
    int frame_width{};
    int frame_height{};
    std::vector<CanonicalPerson> people;
    std::vector<CanonicalEvent> events;
};

void validateContractVersion(std::string_view version);
void validateCanonicalMetadata(const CanonicalFrameResult& result);
std::string canonicalJson(const CanonicalEvent& event);
std::string canonicalJson(const CanonicalFrameResult& result);
std::string runtimeDefaultsJson();

}  // namespace cuajone
