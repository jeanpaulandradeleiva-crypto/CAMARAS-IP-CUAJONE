// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include "cuajone/letterbox.hpp"
#include "cuajone/types.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace cuajone {

enum class TensorLayout {
    ChannelsFirst,
    PredictionsFirst,
};

enum class YoloOutputFormat {
    Raw,
    PoseEndToEnd,
    DetectEndToEnd,
};

struct YoloSchema {
    YoloOutputFormat format{YoloOutputFormat::Raw};
    TensorLayout layout{};
    std::size_t predictions{};
    std::size_t channels{};
    bool has_objectness{};
    std::size_t class_count{};
    std::size_t keypoint_count{};
    std::size_t keypoint_dimensions{};
};

struct TensorView {
    std::span<const float> values;
    std::span<const std::int64_t> shape;
};

struct DecodeLimits {
    std::size_t max_nms_candidates{30000};
    std::size_t max_detections{300};
};

struct OnnxPoseContract {
    std::size_t class_count;
    std::array<int, 2> keypoint_shape;
};

OnnxPoseContract validateOnnxPoseContract(
    std::size_t class_count,
    std::array<int, 2> keypoint_shape);

YoloSchema validateDetectSchema(
    std::span<const std::int64_t> shape,
    std::size_t class_count);
YoloSchema validatePoseSchema(
    std::span<const std::int64_t> shape,
    std::size_t class_count,
    std::size_t keypoint_count,
    std::size_t keypoint_dimensions);

std::vector<Detection> decodeDetections(
    const TensorView& tensor,
    std::size_t class_count,
    std::span<const float> class_confidence_thresholds,
    float iou_threshold,
    const LetterboxTransform& transform,
    DecodeLimits limits = {});
std::vector<Detection> decodeDetections(
    const TensorView& tensor,
    std::size_t class_count,
    std::span<const float> class_confidence_thresholds,
    std::span<const std::uint8_t> class_enabled,
    float iou_threshold,
    const LetterboxTransform& transform,
    DecodeLimits limits = {});
std::vector<Detection> decodeDetections(
    const TensorView& tensor,
    std::size_t class_count,
    float confidence_threshold,
    float iou_threshold,
    const LetterboxTransform& transform,
    DecodeLimits limits = {});
std::vector<PoseDetection> decodePoses(
    const TensorView& tensor,
    std::size_t class_count,
    std::size_t keypoint_count,
    std::size_t keypoint_dimensions,
    float confidence_threshold,
    float iou_threshold,
    const LetterboxTransform& transform,
    DecodeLimits limits = {});

std::vector<Detection> classAwareNms(
    std::vector<Detection> detections,
    float iou_threshold);
std::vector<PoseDetection> classAwarePoseNms(
    std::vector<PoseDetection> detections,
    float iou_threshold);

}  // namespace cuajone
