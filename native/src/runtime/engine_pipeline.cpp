// SPDX-License-Identifier: AGPL-3.0-only

#include "cuajone/engine_pipeline.hpp"

#include "cuajone/onnx_session.hpp"
#include "cuajone/preprocess.hpp"
#include "cuajone/yolo_decode.hpp"

#ifdef CUAJONE_WITH_TENSORRT
#include "cuajone/engine_reader.hpp"
#include "cuajone/tensorrt_runtime.hpp"
#endif

#include <optional>
#include <algorithm>
#include <stdexcept>
#include <utility>

namespace cuajone {
namespace {

void validateContiguousNames(const std::map<int, std::string>& names, const std::string& model_name) {
    if (names.empty()) throw std::runtime_error(model_name + " requires explicit class labels");
    int expected = 0;
    for (const auto& [id, name] : names) {
        if (id != expected++ || name.empty()) {
            throw std::runtime_error(model_name + " class IDs must be contiguous from zero");
        }
    }
}

#ifdef CUAJONE_WITH_TENSORRT
void validateTask(const EngineMetadata& metadata, const std::string& expected, const std::string& engine_name) {
    if (metadata.task && normalizeLabel(*metadata.task) != expected) {
        throw std::runtime_error(engine_name + " metadata task is '" + *metadata.task
            + "', expected '" + expected + "'");
    }
}
#endif

}  // namespace

struct NativeEnginePipeline::Impl {
    explicit Impl(EnginePipelineConfig input)
        : config(std::move(input)), analytics(config.analytics) {
        if (config.maximum_detections == 0
            || config.maximum_detections > DecodeLimits{}.max_nms_candidates) {
            throw std::invalid_argument("maximum_detections is outside the supported range");
        }
        if (config.backend == ComputeBackend::Cpu) loadCpu();
        else if (config.backend == ComputeBackend::Cuda) loadCuda();
        else throw std::invalid_argument("Engine pipeline requires a resolved cpu or cuda backend");
    }

    void loadCpu() {
        summary.backend = ComputeBackend::Cpu;
        summary.provider = "ONNX Runtime CPUExecutionProvider";
        if (!std::filesystem::is_regular_file(config.ppe_onnx)) {
            throw std::runtime_error("PPE ONNX model does not exist: " + config.ppe_onnx.string());
        }
        if (config.analytics.mode == AnalyticsMode::PpeFall
            && !std::filesystem::is_regular_file(config.pose_onnx)) {
            throw std::runtime_error("Pose ONNX model does not exist: " + config.pose_onnx.string());
        }
        if (!config.ppe_labels) {
            throw std::runtime_error("CPU inference requires --ppe-labels for explicit class semantics");
        }
        ppe_names = *config.ppe_labels;
        validateContiguousNames(ppe_names, "PPE ONNX model");
        ppe_classes = resolvePpeClasses(ppe_names);
        ppe_session = std::make_unique<OnnxSession>(config.ppe_onnx, ModelRole::Ppe);
        validateDetectSchema(ppe_session->outputShape(), ppe_names.size());
        ppe_preprocessor = std::make_unique<LetterboxPreprocessor>(
            ppe_session->inputWidth(), ppe_session->inputHeight());

        if (config.analytics.mode == AnalyticsMode::PpeFall) {
            pose_class_count = config.pose_class_count;
            keypoint_shape = config.pose_keypoint_shape;
            if (pose_class_count != 1) {
                throw std::runtime_error("The native pose pipeline supports exactly one person class");
            }
            if (keypoint_shape[1] < 3) {
                throw std::runtime_error("Pose keypoint dimensions must include x, y, and confidence");
            }
            pose_session = std::make_unique<OnnxSession>(config.pose_onnx, ModelRole::Pose);
            validatePoseSchema(
                pose_session->outputShape(), pose_class_count,
                static_cast<std::size_t>(keypoint_shape[0]),
                static_cast<std::size_t>(keypoint_shape[1]));
            pose_preprocessor = std::make_unique<LetterboxPreprocessor>(
                pose_session->inputWidth(), pose_session->inputHeight());
            summary.pose_loaded = true;
        }
    }

    void loadCuda() {
        if (config.ppe_engine.empty()
            && (config.analytics.mode == AnalyticsMode::PpeOnly || config.pose_engine.empty())) {
            loadCudaOnnx();
            return;
        }
#ifdef CUAJONE_WITH_TENSORRT
        summary.backend = ComputeBackend::Cuda;
        summary.provider = "TensorRT 11/CUDA";
        if (!std::filesystem::is_regular_file(config.ppe_engine)) {
            throw std::runtime_error("PPE engine does not exist: " + config.ppe_engine.string());
        }
        if (config.analytics.mode == AnalyticsMode::PpeFall
            && !std::filesystem::is_regular_file(config.pose_engine)) {
            throw std::runtime_error("Pose engine does not exist: " + config.pose_engine.string());
        }
        const DeviceSummary device = selectCudaDevice(config.device);
        summary.device_name = device.name;
        summary.device_index = device.selected;
        summary.device_count = device.count;
        summary.compute_major = device.compute_major;
        summary.compute_minor = device.compute_minor;

        ppe_file.emplace(EngineFile::read(config.ppe_engine));
        validateTask(ppe_file->metadata(), "detect", "PPE engine");
        ppe_names = ppe_file->metadata().names;
        if (ppe_names.empty() && config.ppe_labels) ppe_names = *config.ppe_labels;
        validateContiguousNames(ppe_names, "PPE engine");
        ppe_classes = resolvePpeClasses(ppe_names);
        ppe_session = std::make_unique<TensorRtSession>(*ppe_file, ppe_file->metadata().image_size);
        validateDetectSchema(ppe_session->outputShape(), ppe_names.size());
        ppe_preprocessor = std::make_unique<LetterboxPreprocessor>(
            ppe_session->inputWidth(), ppe_session->inputHeight());
        summary.ppe_metadata_prefix = ppe_file->hasMetadataPrefix();

        if (config.analytics.mode == AnalyticsMode::PpeFall) {
            pose_file.emplace(EngineFile::read(config.pose_engine));
            validateTask(pose_file->metadata(), "pose", "Pose engine");
            pose_class_count = pose_file->metadata().names.empty()
                ? config.pose_class_count : pose_file->metadata().names.size();
            if (!pose_file->metadata().names.empty()) {
                validateContiguousNames(pose_file->metadata().names, "Pose engine");
            }
            if (pose_class_count != 1) {
                throw std::runtime_error("The native pose pipeline supports exactly one person class");
            }
            if (!pose_file->metadata().names.empty()
                && !isPersonClassLabel(pose_file->metadata().names.begin()->second)
                && !config.allow_nonperson_pose_class) {
                throw std::runtime_error(
                    "Pose engine's single metadata class must normalize to person/persona; "
                    "use the explicit override only after verifying the engine contract");
            }
            keypoint_shape = pose_file->metadata().keypoint_shape.value_or(config.pose_keypoint_shape);
            if (keypoint_shape[1] < 3) {
                throw std::runtime_error("Pose keypoint dimensions must include x, y, and confidence");
            }
            pose_session = std::make_unique<TensorRtSession>(*pose_file, pose_file->metadata().image_size);
            validatePoseSchema(
                pose_session->outputShape(), pose_class_count,
                static_cast<std::size_t>(keypoint_shape[0]),
                static_cast<std::size_t>(keypoint_shape[1]));
            pose_preprocessor = std::make_unique<LetterboxPreprocessor>(
                pose_session->inputWidth(), pose_session->inputHeight());
            summary.pose_loaded = true;
            summary.pose_metadata_prefix = pose_file->hasMetadataPrefix();
        }
#else
        throw std::runtime_error("CUDA mode is unavailable in this CPU-only build");
#endif
    }

    void loadCudaOnnx() {
        summary.backend = ComputeBackend::Cuda;
        summary.provider = "ONNX Runtime CUDAExecutionProvider";
        if (!std::filesystem::is_regular_file(config.ppe_onnx)) {
            throw std::runtime_error("PPE ONNX model does not exist: " + config.ppe_onnx.string());
        }
        if (config.analytics.mode == AnalyticsMode::PpeFall
            && !std::filesystem::is_regular_file(config.pose_onnx)) {
            throw std::runtime_error("Pose ONNX model does not exist: " + config.pose_onnx.string());
        }
        if (!config.ppe_labels) {
            throw std::runtime_error("CUDA ONNX inference requires --ppe-labels for explicit class semantics");
        }
        const auto probe = probeHardware();
        if (probe.status != HardwareProbeStatus::CudaReady) {
            throw std::runtime_error("CUDA ONNX provider requires a ready NVIDIA CUDA device");
        }
        const int device = selectCompatibleCudaDevice(probe.cuda_devices, config.device);
        const auto selected = std::find_if(probe.cuda_devices.begin(), probe.cuda_devices.end(), [&](const auto& value) {
            return value.device_index == device;
        });
        summary.device_index = device;
        summary.device_count = static_cast<int>(probe.cuda_devices.size());
        summary.device_name = selected->name;
        summary.compute_major = selected->compute_major;
        summary.compute_minor = selected->compute_minor;
        ppe_names = *config.ppe_labels;
        validateContiguousNames(ppe_names, "PPE ONNX model");
        ppe_classes = resolvePpeClasses(ppe_names);
        ppe_session = std::make_unique<OnnxSession>(config.ppe_onnx, ModelRole::Ppe, OnnxSessionOptions{
            OnnxExecutionProvider::Cuda, device,
        });
        validateDetectSchema(ppe_session->outputShape(), ppe_names.size());
        ppe_preprocessor = std::make_unique<LetterboxPreprocessor>(
            ppe_session->inputWidth(), ppe_session->inputHeight());
        if (config.analytics.mode == AnalyticsMode::PpeFall) {
            if (config.pose_class_count != 1 || config.pose_keypoint_shape[1] < 3) {
                throw std::runtime_error("CUDA ONNX pose contract is unsupported");
            }
            pose_session = std::make_unique<OnnxSession>(config.pose_onnx, ModelRole::Pose, OnnxSessionOptions{
                OnnxExecutionProvider::Cuda, device,
            });
            validatePoseSchema(
                pose_session->outputShape(), config.pose_class_count,
                static_cast<std::size_t>(config.pose_keypoint_shape[0]),
                static_cast<std::size_t>(config.pose_keypoint_shape[1]));
            pose_preprocessor = std::make_unique<LetterboxPreprocessor>(
                pose_session->inputWidth(), pose_session->inputHeight());
            summary.pose_loaded = true;
        }
    }

    ProcessedFrame process(
        const cv::Mat& frame,
        std::string source_id,
        std::uint64_t frame_id,
        std::int64_t monotonic_timestamp_ms,
        std::string observed_at) {
        if (frame.empty() || frame.type() != CV_8UC3) {
            throw std::invalid_argument("Engine pipeline requires a non-empty CV_8UC3 BGR frame");
        }
        const auto ppe_input = ppe_preprocessor->process(frame);
        const auto ppe_output = ppe_session->infer(ppe_input.nchw);
        auto ppe_detections = decodeDetections(
            {ppe_output.values, ppe_output.shape},
            ppe_names.size(), config.ppe_confidence, config.nms_iou,
            ppe_input.transform,
            {DecodeLimits{}.max_nms_candidates, config.maximum_detections});
        std::vector<PoseDetection> poses;
        if (pose_session) {
            const auto pose_input = pose_preprocessor->process(frame);
            const auto pose_output = pose_session->infer(pose_input.nchw);
            poses = decodePoses(
                {pose_output.values, pose_output.shape}, pose_class_count,
                static_cast<std::size_t>(keypoint_shape[0]),
                static_cast<std::size_t>(keypoint_shape[1]),
                config.pose_confidence, config.nms_iou, pose_input.transform,
                {DecodeLimits{}.max_nms_candidates, config.maximum_detections});
        }
        return analytics.process({
            std::string(kContractVersion), std::move(source_id), frame_id,
            monotonic_timestamp_ms, std::move(observed_at), frame.cols, frame.rows,
            std::move(ppe_detections), std::move(poses), ppe_classes,
        });
    }

    EnginePipelineConfig config;
    EnginePipelineSummary summary;
#ifdef CUAJONE_WITH_TENSORRT
    std::optional<EngineFile> ppe_file;
    std::optional<EngineFile> pose_file;
#endif
    std::map<int, std::string> ppe_names;
    PpeClassMap ppe_classes;
    std::size_t pose_class_count{};
    std::array<int, 2> keypoint_shape{};
    std::unique_ptr<InferenceSession> ppe_session;
    std::unique_ptr<InferenceSession> pose_session;
    std::unique_ptr<LetterboxPreprocessor> ppe_preprocessor;
    std::unique_ptr<LetterboxPreprocessor> pose_preprocessor;
    AnalyticsPipeline analytics;
};

NativeEnginePipeline::NativeEnginePipeline(EnginePipelineConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

NativeEnginePipeline::~NativeEnginePipeline() = default;

ProcessedFrame NativeEnginePipeline::processFrame(
    const cv::Mat& bgr_frame,
    std::string source_id,
    std::uint64_t frame_id,
    std::int64_t monotonic_timestamp_ms,
    std::string observed_at) {
    return impl_->process(
        bgr_frame, std::move(source_id), frame_id, monotonic_timestamp_ms,
        std::move(observed_at));
}

void NativeEnginePipeline::reset() noexcept {
    impl_->analytics.reset();
}

const EnginePipelineSummary& NativeEnginePipeline::summary() const noexcept {
    return impl_->summary;
}

bool tensorRtBackendCompiled() noexcept {
#ifdef CUAJONE_WITH_TENSORRT
    return true;
#else
    return false;
#endif
}

bool onnxCudaExecutionProviderCompiled() noexcept {
    return true;
}

}  // namespace cuajone
