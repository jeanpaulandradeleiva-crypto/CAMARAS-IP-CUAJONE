// SPDX-License-Identifier: AGPL-3.0-only

#include "cuajone/preprocess.hpp"

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <stdexcept>

namespace cuajone {

LetterboxPreprocessor::LetterboxPreprocessor(int model_width, int model_height)
    : model_width_(model_width), model_height_(model_height) {
    if (model_width <= 0 || model_height <= 0) {
        throw std::invalid_argument("Model dimensions must be positive");
    }
    packed_.resize(static_cast<std::size_t>(3 * model_width * model_height));
}

PreprocessedFrame LetterboxPreprocessor::process(const cv::Mat& bgr_frame) {
    if (bgr_frame.empty() || bgr_frame.type() != CV_8UC3) {
        throw std::invalid_argument("Preprocessing requires a non-empty CV_8UC3 frame");
    }

    const LetterboxTransform transform = makeLetterboxTransform(
        bgr_frame.cols, bgr_frame.rows, model_width_, model_height_);
    const int resized_width = static_cast<int>(std::round(bgr_frame.cols * transform.scale_x));
    const int resized_height = static_cast<int>(std::round(bgr_frame.rows * transform.scale_y));

    cv::resize(bgr_frame, resized_, cv::Size(resized_width, resized_height), 0.0, 0.0, cv::INTER_LINEAR);
    canvas_.create(model_height_, model_width_, CV_8UC3);
    canvas_.setTo(cv::Scalar(114, 114, 114));
    resized_.copyTo(canvas_(cv::Rect(
        transform.padding_left, transform.padding_top, resized_width, resized_height)));

    const std::size_t plane = static_cast<std::size_t>(model_width_ * model_height_);
    for (int y = 0; y < model_height_; ++y) {
        const auto* row = canvas_.ptr<cv::Vec3b>(y);
        for (int x = 0; x < model_width_; ++x) {
            const std::size_t offset = static_cast<std::size_t>(y * model_width_ + x);
            packed_[offset] = static_cast<float>(row[x][2]) / 255.0F;
            packed_[plane + offset] = static_cast<float>(row[x][1]) / 255.0F;
            packed_[2 * plane + offset] = static_cast<float>(row[x][0]) / 255.0F;
        }
    }

    return {
        packed_,
        transform,
    };
}

}  // namespace cuajone
