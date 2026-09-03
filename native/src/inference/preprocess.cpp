// SPDX-License-Identifier: AGPL-3.0-only

#include "cuajone/preprocess.hpp"
#include "cuajone/resource_limits.hpp"

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <stdexcept>

namespace cuajone {

LetterboxPreprocessor::LetterboxPreprocessor(int model_width, int model_height)
    : model_width_(model_width), model_height_(model_height) {
    if (model_width <= 0 || model_height <= 0) {
        throw std::invalid_argument("Model dimensions must be positive");
    }
    if (model_width > resource_limits::kMaximumImageDimension
        || model_height > resource_limits::kMaximumImageDimension) {
        throw std::invalid_argument("Model dimensions exceed the supported image limit");
    }
}

bool canSharePreprocessedInput(
    int ppe_width, int ppe_height, int pose_width, int pose_height) noexcept {
    return ppe_width > 0 && ppe_height > 0
        && ppe_width == pose_width && ppe_height == pose_height;
}

std::shared_ptr<std::vector<float>> LetterboxPreprocessor::acquirePackedBuffer(
    std::size_t element_count) {
    for (auto& buffer : packed_pool_) {
        // use_count == 1 means only the pool itself still owns the buffer, so
        // no async consumer can still be reading it.
        if (buffer.use_count() == 1) {
            buffer->resize(element_count);
            return buffer;
        }
    }
    auto buffer = std::make_shared<std::vector<float>>(element_count);
    packed_pool_.push_back(buffer);
    return buffer;
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

    const std::size_t width = static_cast<std::size_t>(model_width_);
    const std::size_t plane = width * static_cast<std::size_t>(model_height_);
    auto packed = acquirePackedBuffer(resource_limits::checkedMultiply(
        3, plane, resource_limits::kMaximumInputElements, "Preprocessed input"));
    float* red = packed->data();
    float* green = red + plane;
    float* blue = green + plane;
    for (int y = 0; y < model_height_; ++y) {
        const auto* row = canvas_.ptr<cv::Vec3b>(y);
        for (int x = 0; x < model_width_; ++x) {
            const auto& pixel = row[x];
            *red++ = static_cast<float>(pixel[2]) / 255.0F;
            *green++ = static_cast<float>(pixel[1]) / 255.0F;
            *blue++ = static_cast<float>(pixel[0]) / 255.0F;
        }
    }

    return {
        std::move(packed),
        transform,
    };
}

}  // namespace cuajone
