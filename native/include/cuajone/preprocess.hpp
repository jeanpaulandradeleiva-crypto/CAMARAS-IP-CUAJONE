// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include "cuajone/letterbox.hpp"

#include <opencv2/core/mat.hpp>

#include <memory>
#include <span>
#include <vector>

namespace cuajone {

struct PreprocessedFrame {
    std::shared_ptr<const std::vector<float>> packed;
    LetterboxTransform transform;

    [[nodiscard]] std::span<const float> nchw() const noexcept { return *packed; }
};

[[nodiscard]] bool canSharePreprocessedInput(
    int ppe_width, int ppe_height, int pose_width, int pose_height) noexcept;

class LetterboxPreprocessor {
public:
    LetterboxPreprocessor(int model_width, int model_height);

    PreprocessedFrame process(const cv::Mat& bgr_frame);

private:
    // Returns a buffer from a small internal pool, reusing one only when no
    // async consumer (e.g. the pose executor) still holds a reference. This
    // removes the per-frame packed-tensor allocation from the hot path.
    std::shared_ptr<std::vector<float>> acquirePackedBuffer(std::size_t element_count);

    int model_width_;
    int model_height_;
    cv::Mat resized_;
    cv::Mat canvas_;
    std::vector<std::shared_ptr<std::vector<float>>> packed_pool_;
};

}  // namespace cuajone
