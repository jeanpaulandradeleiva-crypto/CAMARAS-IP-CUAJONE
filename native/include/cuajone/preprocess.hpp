// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include "cuajone/letterbox.hpp"

#include <opencv2/core/mat.hpp>

#include <span>
#include <vector>

namespace cuajone {

struct PreprocessedFrame {
    std::span<const float> nchw;
    LetterboxTransform transform;
};

class LetterboxPreprocessor {
public:
    LetterboxPreprocessor(int model_width, int model_height);

    PreprocessedFrame process(const cv::Mat& bgr_frame);

private:
    int model_width_;
    int model_height_;
    cv::Mat resized_;
    cv::Mat canvas_;
    std::vector<float> packed_;
};

}  // namespace cuajone
