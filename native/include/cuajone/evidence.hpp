// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include "cuajone/types.hpp"

#include <opencv2/core/mat.hpp>

#include <filesystem>
#include <string>

namespace cuajone {

struct EvidenceRecord {
    std::string timestamp;
    std::string source_label;
    int track_id{};
    std::string event_type;
    float confidence{};
    std::string status;
    std::filesystem::path image_path;
};

void validateWritableOutput(const std::filesystem::path& output);

class EvidenceWriter {
public:
    explicit EvidenceWriter(std::filesystem::path output);
    EvidenceRecord append(
        const cv::Mat& annotated_frame,
        const std::string& source_label,
        const EventCandidate& event);

private:
    std::filesystem::path output_;
    std::filesystem::path images_;
    std::filesystem::path csv_;
    std::uint64_t sequence_{};
};

}  // namespace cuajone
