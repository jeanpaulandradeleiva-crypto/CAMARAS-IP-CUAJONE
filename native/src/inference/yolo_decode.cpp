// SPDX-License-Identifier: AGPL-3.0-only

#include "cuajone/yolo_decode.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>
#include <queue>
#include <stdexcept>
#include <tuple>

namespace cuajone {
namespace {

struct Shape2D {
    std::size_t first;
    std::size_t second;
};

Shape2D predictionShape(std::span<const std::int64_t> shape) {
    if (shape.size() == 3) {
        if (shape[0] != 1) {
            throw std::runtime_error("YOLO output must have a fixed batch size of 1");
        }
        shape = shape.subspan(1);
    }
    if (shape.size() != 2 || shape[0] <= 0 || shape[1] <= 0) {
        throw std::runtime_error("YOLO output must be rank 2 or rank 3 with positive fixed dimensions");
    }
    return {static_cast<std::size_t>(shape[0]), static_cast<std::size_t>(shape[1])};
}

YoloSchema validateSchema(
    std::span<const std::int64_t> shape,
    std::size_t class_count,
    std::size_t keypoint_count,
    std::size_t keypoint_dimensions) {
    if (class_count == 0) {
        throw std::invalid_argument("YOLO decoding requires at least one class");
    }
    if (keypoint_count > 0 && keypoint_dimensions < 3) {
        throw std::invalid_argument("Pose keypoints require x, y, and confidence dimensions");
    }

    const Shape2D dimensions = predictionShape(shape);
    const std::size_t payload = class_count + keypoint_count * keypoint_dimensions;
    const std::size_t without_objectness = 4 + payload;
    const std::size_t with_objectness = 5 + payload;

    struct Candidate {
        TensorLayout layout;
        std::size_t predictions;
        std::size_t channels;
        bool objectness;
    };
    std::vector<Candidate> candidates;
    const auto addCandidate = [&](std::size_t channels, std::size_t predictions, TensorLayout layout) {
        if (channels == without_objectness || channels == with_objectness) {
            candidates.push_back({layout, predictions, channels, channels == with_objectness});
        }
    };
    addCandidate(dimensions.first, dimensions.second, TensorLayout::ChannelsFirst);
    addCandidate(dimensions.second, dimensions.first, TensorLayout::PredictionsFirst);

    if (candidates.empty()) {
        throw std::runtime_error(
            "Unsupported raw YOLO output: neither dimension matches the expected channel formulas "
            "4+classes+keypoints nor 5+classes+keypoints");
    }
    if (candidates.size() != 1) {
        throw std::runtime_error("Ambiguous raw YOLO output layout; both dimensions match a supported channel formula");
    }
    const auto& candidate = candidates.front();
    return {
        YoloOutputFormat::Raw,
        candidate.layout,
        candidate.predictions,
        candidate.channels,
        candidate.objectness,
        class_count,
        keypoint_count,
        keypoint_dimensions,
    };
}

std::optional<YoloSchema> validateApprovedEndToEndPoseSchema(
    std::span<const std::int64_t> shape,
    std::size_t class_count,
    std::size_t keypoint_count,
    std::size_t keypoint_dimensions) {
    if (shape.size() != 3 || shape[0] != 1 || shape[1] != 300) return std::nullopt;
    const std::size_t expected_channels = 6 + keypoint_count * keypoint_dimensions;
    if (shape[2] != static_cast<std::int64_t>(expected_channels)) return std::nullopt;
    if (class_count == 0 || keypoint_count == 0 || keypoint_dimensions < 3) {
        throw std::invalid_argument("End-to-end pose decoding requires classes and x/y/confidence keypoints");
    }
    return YoloSchema{
        YoloOutputFormat::PoseEndToEnd,
        TensorLayout::PredictionsFirst,
        300,
        expected_channels,
        false,
        class_count,
        keypoint_count,
        keypoint_dimensions,
    };
}

class PredictionAccessor {
public:
    PredictionAccessor(const TensorView& tensor, YoloSchema schema)
        : tensor_(tensor), schema_(schema) {
        const std::size_t expected = schema_.predictions * schema_.channels;
        if (tensor_.values.size() != expected) {
            throw std::runtime_error("YOLO tensor data length does not match its declared shape");
        }
    }

    float at(std::size_t prediction, std::size_t channel) const {
        if (schema_.layout == TensorLayout::ChannelsFirst) {
            return tensor_.values[channel * schema_.predictions + prediction];
        }
        return tensor_.values[prediction * schema_.channels + channel];
    }

private:
    const TensorView& tensor_;
    YoloSchema schema_;
};

struct Confidence {
    float value;
    int class_id;
};

std::optional<Confidence> confidenceFor(
    const PredictionAccessor& values,
    const YoloSchema& schema,
    std::size_t prediction) {
    const std::size_t class_offset = schema.has_objectness ? 5 : 4;
    float best_class_score = -std::numeric_limits<float>::infinity();
    int best_class = -1;
    for (std::size_t class_id = 0; class_id < schema.class_count; ++class_id) {
        const float score = values.at(prediction, class_offset + class_id);
        if (!std::isfinite(score)) return std::nullopt;
        if (score > best_class_score) {
            best_class_score = score;
            best_class = static_cast<int>(class_id);
        }
    }
    if (best_class < 0) {
        return std::nullopt;
    }
    const float objectness = schema.has_objectness ? values.at(prediction, 4) : 1.0F;
    if (!std::isfinite(objectness)) return std::nullopt;
    const float confidence = objectness * best_class_score;
    if (!std::isfinite(confidence)) return std::nullopt;
    return Confidence{confidence, best_class};
}

struct RawBox {
    float center_x;
    float center_y;
    float width;
    float height;
};

std::optional<RawBox> rawBox(
    const PredictionAccessor& values,
    std::size_t prediction) {
    const float center_x = values.at(prediction, 0);
    const float center_y = values.at(prediction, 1);
    const float width = values.at(prediction, 2);
    const float height = values.at(prediction, 3);
    if (!std::isfinite(center_x) || !std::isfinite(center_y) || !std::isfinite(width)
        || !std::isfinite(height)) {
        return std::nullopt;
    }
    return RawBox{center_x, center_y, width, height};
}

Box modelBox(const RawBox& raw) {
    return {
        raw.center_x - raw.width / 2.0F,
        raw.center_y - raw.height / 2.0F,
        raw.center_x + raw.width / 2.0F,
        raw.center_y + raw.height / 2.0F,
    };
}

template <typename DetectionType>
bool isFiniteDetection(const DetectionType& detection) {
    const bool finite = std::isfinite(detection.confidence)
        && std::isfinite(detection.box.x1) && std::isfinite(detection.box.y1)
        && std::isfinite(detection.box.x2) && std::isfinite(detection.box.y2);
    if (!finite) return false;
    if constexpr (requires { detection.keypoints; }) {
        return std::all_of(
            detection.keypoints.begin(), detection.keypoints.end(), [](const Keypoint& keypoint) {
                return std::isfinite(keypoint.x) && std::isfinite(keypoint.y)
                    && std::isfinite(keypoint.confidence);
            });
    }
    return true;
}

bool hasFinitePositiveArea(const Box& box) {
    const float area = boxArea(box);
    return std::isfinite(area) && area > 0.0F;
}

struct DetectionBetter {
    template <typename DetectionType>
    bool operator()(const DetectionType& lhs, const DetectionType& rhs) const {
        return std::tuple{-lhs.confidence, lhs.class_id, lhs.box.x1, lhs.box.y1,
                          lhs.box.x2, lhs.box.y2}
            < std::tuple{-rhs.confidence, rhs.class_id, rhs.box.x1, rhs.box.y1,
                         rhs.box.x2, rhs.box.y2};
    }
};

template <typename DetectionType>
class BoundedCandidates {
public:
    explicit BoundedCandidates(std::size_t limit) : limit_(limit) {}

    void add(DetectionType candidate) {
        if (candidates_.size() < limit_) {
            candidates_.push(std::move(candidate));
        } else if (DetectionBetter{}(candidate, candidates_.top())) {
            candidates_.pop();
            candidates_.push(std::move(candidate));
        }
    }

    std::vector<DetectionType> take() {
        std::vector<DetectionType> result;
        result.reserve(candidates_.size());
        while (!candidates_.empty()) {
            result.push_back(candidates_.top());
            candidates_.pop();
        }
        return result;
    }

private:
    std::size_t limit_;
    std::priority_queue<DetectionType, std::vector<DetectionType>, DetectionBetter> candidates_;
};

void validateDecodeOptions(
    float confidence_threshold,
    float iou_threshold,
    const LetterboxTransform& transform,
    const DecodeLimits& limits) {
    if (!std::isfinite(confidence_threshold) || confidence_threshold < 0.0F
        || confidence_threshold > 1.0F) {
        throw std::invalid_argument("YOLO confidence threshold must be finite and in [0, 1]");
    }
    if (!std::isfinite(iou_threshold) || iou_threshold < 0.0F || iou_threshold > 1.0F) {
        throw std::invalid_argument("NMS IoU threshold must be finite and in [0, 1]");
    }
    if (limits.max_nms_candidates == 0 || limits.max_detections == 0
        || limits.max_detections > limits.max_nms_candidates) {
        throw std::invalid_argument("YOLO decode limits must be positive and max detections must not exceed the pre-NMS limit");
    }
    if (transform.source_width <= 0 || transform.source_height <= 0
        || transform.model_width <= 0 || transform.model_height <= 0
        || !std::isfinite(transform.scale_x) || transform.scale_x <= 0.0F
        || !std::isfinite(transform.scale_y) || transform.scale_y <= 0.0F) {
        throw std::invalid_argument("Letterbox transform must contain positive finite geometry");
    }
}

template <typename DetectionType>
std::vector<DetectionType> nms(
    std::vector<DetectionType> detections,
    float iou_threshold,
    std::size_t max_detections = std::numeric_limits<std::size_t>::max()) {
    if (!std::isfinite(iou_threshold) || iou_threshold < 0.0F || iou_threshold > 1.0F) {
        throw std::invalid_argument("NMS IoU threshold must be finite and in [0, 1]");
    }
    std::erase_if(detections, [](const DetectionType& detection) {
        return !isFiniteDetection(detection) || !hasFinitePositiveArea(detection.box);
    });
    std::stable_sort(detections.begin(), detections.end(), DetectionBetter{});
    std::vector<DetectionType> kept;
    for (auto& candidate : detections) {
        const bool suppressed = std::any_of(kept.begin(), kept.end(), [&](const auto& accepted) {
            return candidate.class_id == accepted.class_id
                && intersectionOverUnion(candidate.box, accepted.box) > iou_threshold;
        });
        if (!suppressed) {
            kept.push_back(std::move(candidate));
            if (kept.size() == max_detections) break;
        }
    }
    return kept;
}

}  // namespace

OnnxPoseContract validateOnnxPoseContract(
    std::size_t class_count,
    std::array<int, 2> keypoint_shape) {
    if (class_count != 1 || keypoint_shape[0] <= 0 || keypoint_shape[1] < 3) {
        throw std::runtime_error("ONNX pose contract is unsupported");
    }
    return {class_count, keypoint_shape};
}

YoloSchema validateDetectSchema(std::span<const std::int64_t> shape, std::size_t class_count) {
    return validateSchema(shape, class_count, 0, 0);
}

YoloSchema validatePoseSchema(
    std::span<const std::int64_t> shape,
    std::size_t class_count,
    std::size_t keypoint_count,
    std::size_t keypoint_dimensions) {
    if (keypoint_count == 0) {
        throw std::invalid_argument("Pose decoding requires at least one keypoint");
    }
    if (const auto end_to_end = validateApprovedEndToEndPoseSchema(
            shape, class_count, keypoint_count, keypoint_dimensions)) {
        return *end_to_end;
    }
    return validateSchema(shape, class_count, keypoint_count, keypoint_dimensions);
}

std::vector<Detection> decodeDetections(
    const TensorView& tensor,
    std::size_t class_count,
    float confidence_threshold,
    float iou_threshold,
    const LetterboxTransform& transform,
    DecodeLimits limits) {
    validateDecodeOptions(confidence_threshold, iou_threshold, transform, limits);
    const YoloSchema schema = validateDetectSchema(tensor.shape, class_count);
    const PredictionAccessor values(tensor, schema);
    BoundedCandidates<Detection> candidates(limits.max_nms_candidates);
    for (std::size_t prediction = 0; prediction < schema.predictions; ++prediction) {
        const auto raw_box = rawBox(values, prediction);
        const auto confidence = confidenceFor(values, schema, prediction);
        if (!raw_box || !confidence || confidence->value < confidence_threshold
            || raw_box->width <= 0.0F || raw_box->height <= 0.0F) continue;
        const Box box = modelBox(*raw_box);
        const Detection candidate{box, confidence->value, confidence->class_id};
        if (!isFiniteDetection(candidate) || !hasFinitePositiveArea(box)) {
            continue;
        }
        candidates.add(candidate);
    }
    auto detections = nms(candidates.take(), iou_threshold, limits.max_detections);
    for (auto& detection : detections) detection.box = transform.restore(detection.box);
    std::erase_if(detections, [](const Detection& detection) {
        return !hasFinitePositiveArea(detection.box);
    });
    return detections;
}

std::vector<PoseDetection> decodePoses(
    const TensorView& tensor,
    std::size_t class_count,
    std::size_t keypoint_count,
    std::size_t keypoint_dimensions,
    float confidence_threshold,
    float iou_threshold,
    const LetterboxTransform& transform,
    DecodeLimits limits) {
    validateDecodeOptions(confidence_threshold, iou_threshold, transform, limits);
    const YoloSchema schema = validatePoseSchema(
        tensor.shape, class_count, keypoint_count, keypoint_dimensions);
    const PredictionAccessor values(tensor, schema);
    BoundedCandidates<PoseDetection> candidates(limits.max_nms_candidates);
    const bool end_to_end = schema.format == YoloOutputFormat::PoseEndToEnd;
    const std::size_t keypoint_offset = end_to_end
        ? 6 : (schema.has_objectness ? 5 : 4) + class_count;
    for (std::size_t prediction = 0; prediction < schema.predictions; ++prediction) {
        std::optional<RawBox> raw_box;
        std::optional<Confidence> confidence;
        Box decoded_box;
        if (end_to_end) {
            decoded_box = {
                values.at(prediction, 0), values.at(prediction, 1),
                values.at(prediction, 2), values.at(prediction, 3),
            };
            const float score = values.at(prediction, 4);
            const float class_value = values.at(prediction, 5);
            if (std::isfinite(score) && std::isfinite(class_value)
                && class_value == std::floor(class_value)
                && class_value >= 0.0F
                && class_value < static_cast<float>(class_count)) {
                confidence = Confidence{score, static_cast<int>(class_value)};
            }
        } else {
            raw_box = rawBox(values, prediction);
            confidence = confidenceFor(values, schema, prediction);
            if (raw_box) decoded_box = modelBox(*raw_box);
        }
        PoseDetection detection;
        bool finite_keypoints = true;
        detection.keypoints.reserve(keypoint_count);
        for (std::size_t keypoint = 0; keypoint < keypoint_count; ++keypoint) {
            const std::size_t offset = keypoint_offset + keypoint * keypoint_dimensions;
            const float x = values.at(prediction, offset);
            const float y = values.at(prediction, offset + 1);
            const float point_confidence = values.at(prediction, offset + 2);
            if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(point_confidence)) {
                finite_keypoints = false;
                break;
            }
            detection.keypoints.push_back({x, y, point_confidence});
        }
        if ((!end_to_end && !raw_box) || !confidence || !finite_keypoints
            || confidence->value < confidence_threshold
            || (!end_to_end && (raw_box->width <= 0.0F || raw_box->height <= 0.0F))) continue;
        detection.box = decoded_box;
        detection.confidence = confidence->value;
        detection.class_id = confidence->class_id;
        if (!isFiniteDetection(detection) || !hasFinitePositiveArea(detection.box)) continue;
        candidates.add(std::move(detection));
    }
    auto detections = candidates.take();
    if (end_to_end) {
        std::stable_sort(detections.begin(), detections.end(), DetectionBetter{});
        if (detections.size() > limits.max_detections) detections.resize(limits.max_detections);
    } else {
        detections = nms(std::move(detections), iou_threshold, limits.max_detections);
    }
    for (auto& detection : detections) {
        detection.box = transform.restore(detection.box);
        for (auto& keypoint : detection.keypoints) {
            const Point restored = transform.restore(Point{keypoint.x, keypoint.y});
            keypoint.x = restored.x;
            keypoint.y = restored.y;
        }
    }
    std::erase_if(detections, [](const PoseDetection& detection) {
        return !hasFinitePositiveArea(detection.box);
    });
    return detections;
}

std::vector<Detection> classAwareNms(std::vector<Detection> detections, float iou_threshold) {
    return nms(std::move(detections), iou_threshold);
}

std::vector<PoseDetection> classAwarePoseNms(
    std::vector<PoseDetection> detections,
    float iou_threshold) {
    return nms(std::move(detections), iou_threshold);
}

}  // namespace cuajone
