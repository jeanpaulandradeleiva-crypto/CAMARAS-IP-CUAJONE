// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include "cuajone/analytics_pipeline.hpp"
#include "cuajone/compute.hpp"
#include "cuajone/inference_settings.hpp"

#include <opencv2/core/mat.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string>

namespace cuajone {

struct EnginePipelineConfig {
    ComputeBackend backend{ComputeBackend::Cuda};
    InferenceProvider provider{InferenceProvider::TensorRt};
    std::filesystem::path ppe_engine;
    std::filesystem::path pose_engine;
    std::filesystem::path ppe_onnx;
    std::filesystem::path pose_onnx;
    std::optional<std::map<int, std::string>> ppe_labels;
    std::size_t pose_class_count{1};
    std::array<int, 2> pose_keypoint_shape{17, 3};
    bool allow_nonperson_pose_class{};
    std::optional<int> device;
    int image_size{kDefaultImageSize};
    float ppe_confidence{0.30F};
    std::array<float, kPpeOutputLabels.size()> ppe_class_confidences{
        0.30F, 0.30F, 0.30F, 0.30F, 0.30F, 0.30F, 0.30F, 0.30F,
    };
    float pose_confidence{0.35F};
    float nms_iou{0.45F};
    std::size_t maximum_detections{300};
    AnalyticsPipelineConfig analytics;
};

struct EnginePipelineSummary {
    ComputeBackend backend{ComputeBackend::Cpu};
    std::string provider;
    std::string device_name;
    int device_index{};
    int device_count{};
    int compute_major{};
    int compute_minor{};
    bool ppe_metadata_prefix{};
    bool pose_loaded{};
    bool pose_metadata_prefix{};
    int image_size{kDefaultImageSize};
};

class NativeEnginePipeline {
public:
    explicit NativeEnginePipeline(EnginePipelineConfig config);
    ~NativeEnginePipeline();
    NativeEnginePipeline(const NativeEnginePipeline&) = delete;
    NativeEnginePipeline& operator=(const NativeEnginePipeline&) = delete;

    ProcessedFrame processFrame(
        const cv::Mat& bgr_frame,
        std::string source_id,
        std::uint64_t frame_id,
        std::int64_t monotonic_timestamp_ms,
        std::string observed_at);
    void reset() noexcept;
    [[nodiscard]] const EnginePipelineSummary& summary() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

bool tensorRtBackendCompiled() noexcept;
bool onnxCudaExecutionProviderCompiled() noexcept;

}  // namespace cuajone
