// SPDX-License-Identifier: AGPL-3.0-only

#include "cuajone/engine_pipeline.hpp"

#include "cuajone/onnx_session.hpp"
#include "cuajone/performance_telemetry.hpp"
#include "cuajone/preprocess.hpp"
#include "cuajone/yolo_decode.hpp"

#ifdef CUAJONE_WITH_TENSORRT
#include "cuajone/engine_reader.hpp"
#include "cuajone/tensorrt_runtime.hpp"
#endif

#include <optional>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <stdexcept>
#include <thread>
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

void validateManifestLabels(const OnnxModelManifest& manifest, const std::map<int, std::string>& names) {
    std::vector<std::string> ordered;
    ordered.reserve(names.size());
    for (const auto& [id, name] : names) {
        static_cast<void>(id);
        ordered.push_back(name);
    }
    if (manifest.label_contract != "always-all-seven-v2" || manifest.labels != ordered) {
        throw std::runtime_error("PPE ONNX manifest labels do not match the configured semantic order");
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

class PoseExecutor {
public:
    PoseExecutor() : worker_([this](std::stop_token stop) { run(stop); }) {}
    ~PoseExecutor() { shutdown(); }

    PoseExecutor(const PoseExecutor&) = delete;
    PoseExecutor& operator=(const PoseExecutor&) = delete;

    template <typename Task>
    std::future<std::vector<PoseDetection>> submit(Task&& task) {
        auto promise = std::make_shared<std::promise<std::vector<PoseDetection>>>();
        auto future = promise->get_future();
        {
            std::scoped_lock lock(mutex_);
            if (!accepting_) throw std::runtime_error("Pose executor is shutting down");
            if (task_) throw std::logic_error("Pose executor already has a pending task");
            task_ = [promise, task = std::forward<Task>(task)]() mutable {
                try {
                    promise->set_value(task());
                } catch (...) {
                    promise->set_exception(std::current_exception());
                }
            };
        }
        wake_.notify_one();
        return future;
    }

    void shutdown() noexcept {
        {
            std::scoped_lock lock(mutex_);
            accepting_ = false;
        }
        worker_.request_stop();
        wake_.notify_one();
        if (worker_.joinable()) worker_.join();
    }

private:
    void run(std::stop_token stop) {
        std::unique_lock lock(mutex_);
        for (;;) {
            wake_.wait(lock, [&] { return stop.stop_requested() || task_ != nullptr; });
            if (task_ == nullptr) return;
            auto task = std::move(task_);
            task_ = nullptr;
            lock.unlock();
            task();
            lock.lock();
        }
    }

    std::mutex mutex_;
    std::condition_variable wake_;
    std::function<void()> task_;
    bool accepting_{true};
    std::jthread worker_;
};

}  // namespace

struct NativeEnginePipeline::Impl {
    explicit Impl(EnginePipelineConfig input)
        : config(std::move(input)), analytics(config.analytics) {
        validateImageSize(config.image_size);
        validatePpeClassConfidences(config.ppe_class_confidences);
        summary.image_size = config.image_size;
        summary.pose_requires_person = config.pose_requires_person;
        if (config.maximum_detections == 0
            || config.maximum_detections > DecodeLimits{}.max_nms_candidates) {
            throw std::invalid_argument("maximum_detections is outside the supported range");
        }
        if (config.backend == ComputeBackend::Cpu
            && config.provider == InferenceProvider::OnnxRuntimeCpu) {
            loadCpu();
        } else if (config.backend == ComputeBackend::Cuda
            && config.provider == InferenceProvider::TensorRt) {
            loadCuda();
        } else if (config.backend == ComputeBackend::Cuda
            && config.provider == InferenceProvider::OnnxRuntimeCuda) {
            loadCudaOnnx();
        } else {
            throw std::invalid_argument("Engine pipeline requires a resolved backend and matching provider");
        }
        cachePpeClassEnabledMask();
        if (config.telemetry != nullptr) {
            config.telemetry->setExecutionPath({
                hybrid_pose_executor,
                tensorrt_gpu_overlap,
                shared_preprocessing,
                std::string(computeBackendName(summary.backend)),
                summary.provider,
                summary.device_name,
                summary.device_index,
                summary.device_count,
                summary.compute_major,
                summary.compute_minor,
            });
        }
    }

    ~Impl() { pose_executor.shutdown(); }

    void resolveSharedPreprocessing() noexcept {
        shared_preprocessing = pose_session != nullptr && canSharePreprocessedInput(
            ppe_session->inputWidth(), ppe_session->inputHeight(),
            pose_session->inputWidth(), pose_session->inputHeight());
#ifdef CUAJONE_INTERNAL_DIAGNOSTICS
        shared_preprocessing = shared_preprocessing && !config.force_separate_hybrid_preprocessing;
#endif
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
        ppe_session = std::make_unique<OnnxSession>(
            config.ppe_onnx, ModelRole::Ppe, OnnxSessionOptions{}, config.image_size);
        validateManifestLabels(static_cast<OnnxSession&>(*ppe_session).manifest(), ppe_names);
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
            pose_session = std::make_unique<OnnxSession>(
                config.pose_onnx, ModelRole::Pose, OnnxSessionOptions{}, config.image_size);
            validatePoseSchema(
                pose_session->outputShape(), pose_class_count,
                static_cast<std::size_t>(keypoint_shape[0]),
                static_cast<std::size_t>(keypoint_shape[1]));
            pose_preprocessor = std::make_unique<LetterboxPreprocessor>(
                pose_session->inputWidth(), pose_session->inputHeight());
            summary.pose_loaded = true;
            resolveSharedPreprocessing();
        }
    }

    void loadCuda() {
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
        ppe_session = std::make_unique<TensorRtSession>(
            *ppe_file, std::array{config.image_size, config.image_size});
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
            pose_session = std::make_unique<TensorRtSession>(
                *pose_file, std::array{config.image_size, config.image_size});
            validatePoseSchema(
                pose_session->outputShape(), pose_class_count,
                static_cast<std::size_t>(keypoint_shape[0]),
                static_cast<std::size_t>(keypoint_shape[1]));
            pose_preprocessor = std::make_unique<LetterboxPreprocessor>(
                pose_session->inputWidth(), pose_session->inputHeight());
            summary.pose_loaded = true;
            summary.pose_metadata_prefix = pose_file->hasMetadataPrefix();
            resolveSharedPreprocessing();
            // GPU overlap only helps when the device has enough SMs to co-schedule
            // two FP32 engines; small cards (e.g. GTX 1650 Ti, SM 7.5) serialize at
            // the device level and overlap regresses (~0.8%) with misleading telemetry.
            // The person gate needs the PPE decision before scheduling pose, which is
            // incompatible with submitting both engines up front.
            tensorrt_gpu_overlap = config.analytics.mode == AnalyticsMode::PpeFall
                && device.compute_major >= 8 && !config.pose_requires_person;
#ifdef CUAJONE_INTERNAL_DIAGNOSTICS
            tensorrt_gpu_overlap = tensorrt_gpu_overlap && !config.force_serial_tensorrt;
#endif
        }
#else
        throw std::runtime_error("CUDA mode is unavailable in this CPU-only build");
#endif
    }

    void loadCudaOnnx() {
        summary.backend = ComputeBackend::Cuda;
        summary.provider = config.analytics.mode == AnalyticsMode::PpeFall
            ? "ONNX Runtime CUDAExecutionProvider (PPE) + CPUExecutionProvider (pose)"
            : "ONNX Runtime CUDAExecutionProvider";
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
        }, config.image_size);
        validateManifestLabels(static_cast<OnnxSession&>(*ppe_session).manifest(), ppe_names);
        validateDetectSchema(ppe_session->outputShape(), ppe_names.size());
        ppe_preprocessor = std::make_unique<LetterboxPreprocessor>(
            ppe_session->inputWidth(), ppe_session->inputHeight());
        if (config.analytics.mode == AnalyticsMode::PpeFall) {
            pose_class_count = config.pose_class_count;
            keypoint_shape = config.pose_keypoint_shape;
            const auto pose_contract = validateOnnxPoseContract(pose_class_count, keypoint_shape);
            pose_session = std::make_unique<OnnxSession>(
                config.pose_onnx, ModelRole::Pose,
                OnnxSessionOptions{OnnxExecutionProvider::Cpu, std::nullopt, 1}, config.image_size);
            validatePoseSchema(
                pose_session->outputShape(), pose_contract.class_count,
                static_cast<std::size_t>(pose_contract.keypoint_shape[0]),
                static_cast<std::size_t>(pose_contract.keypoint_shape[1]));
            pose_preprocessor = std::make_unique<LetterboxPreprocessor>(
                pose_session->inputWidth(), pose_session->inputHeight());
            summary.pose_loaded = true;
            resolveSharedPreprocessing();
            // The person gate must wait for the PPE decode, so it cannot use the
            // overlap path that submits pose before any detection is known.
            hybrid_pose_executor = !config.pose_requires_person;
#ifdef CUAJONE_INTERNAL_DIAGNOSTICS
            hybrid_pose_executor = !config.force_serial_hybrid;
#endif
        }
    }

    ProcessedFrame process(
        const cv::Mat& frame,
        std::string source_id,
        std::uint64_t frame_id,
        std::int64_t monotonic_timestamp_ms,
        std::string observed_at) {
        const auto mutex_wait_started = config.telemetry == nullptr
            ? std::chrono::steady_clock::time_point{} : std::chrono::steady_clock::now();
        std::unique_lock<std::mutex> process_lock(process_mutex);
        if (config.telemetry != nullptr) {
            config.telemetry->addSample(PerformanceStage::ProcessMutexWait,
                std::chrono::steady_clock::now() - mutex_wait_started);
        }
        if (frame.empty() || frame.type() != CV_8UC3) {
            throw std::invalid_argument("Engine pipeline requires a non-empty CV_8UC3 BGR frame");
        }
        const auto pipeline_started = config.telemetry == nullptr
            ? std::chrono::steady_clock::time_point{} : std::chrono::steady_clock::now();
        const auto ppe_preprocess_started = config.telemetry == nullptr
            ? std::chrono::steady_clock::time_point{} : std::chrono::steady_clock::now();
        const auto ppe_input = ppe_preprocessor->process(frame);
        if (config.telemetry != nullptr) {
            config.telemetry->addSample(PerformanceStage::PpePreprocess,
                std::chrono::steady_clock::now() - ppe_preprocess_started);
        }
        std::future<std::vector<PoseDetection>> pose_future;
        if (hybrid_pose_executor) {
            PreprocessedFrame pose_input = ppe_input;
            if (!shared_preprocessing) {
                const auto pose_preprocess_started = config.telemetry == nullptr
                    ? std::chrono::steady_clock::time_point{} : std::chrono::steady_clock::now();
                pose_input = pose_preprocessor->process(frame);
                if (config.telemetry != nullptr) {
                    config.telemetry->addSample(PerformanceStage::PosePreprocess,
                        std::chrono::steady_clock::now() - pose_preprocess_started);
                }
            }
            pose_future = pose_executor.submit([this, pose_input = std::move(pose_input)]() mutable {
                const auto pose_inference_started = config.telemetry == nullptr
                    ? std::chrono::steady_clock::time_point{} : std::chrono::steady_clock::now();
                const auto pose_output = pose_session->infer(pose_input.nchw());
                if (config.telemetry != nullptr) {
                    config.telemetry->addSample(PerformanceStage::PoseInference,
                        std::chrono::steady_clock::now() - pose_inference_started);
                }
                const auto pose_decode_started = config.telemetry == nullptr
                    ? std::chrono::steady_clock::time_point{} : std::chrono::steady_clock::now();
                auto poses = decodePoses(
                    {pose_output.values, pose_output.shape}, pose_class_count,
                    static_cast<std::size_t>(keypoint_shape[0]),
                    static_cast<std::size_t>(keypoint_shape[1]),
                    config.analytics.tracker.low_confidence_threshold,
                    config.nms_iou, pose_input.transform,
                    {DecodeLimits{}.max_nms_candidates, config.maximum_detections});
                if (config.telemetry != nullptr) {
                    config.telemetry->addSample(PerformanceStage::PoseDecode,
                        std::chrono::steady_clock::now() - pose_decode_started);
                }
                return poses;
            });
        }
        std::vector<Detection> ppe_detections;
        std::vector<PoseDetection> poses;
        if (tensorrt_gpu_overlap) {
#ifdef CUAJONE_WITH_TENSORRT
            auto& ppe_trt = static_cast<TensorRtSession&>(*ppe_session);
            auto& pose_trt = static_cast<TensorRtSession&>(*pose_session);
            PreprocessedFrame pose_input = ppe_input;
            if (!shared_preprocessing) {
                const auto pose_preprocess_started = config.telemetry == nullptr
                    ? std::chrono::steady_clock::time_point{} : std::chrono::steady_clock::now();
                pose_input = pose_preprocessor->process(frame);
                if (config.telemetry != nullptr) {
                    config.telemetry->addSample(PerformanceStage::PosePreprocess,
                        std::chrono::steady_clock::now() - pose_preprocess_started);
                }
            }
            ppe_trt.submit(ppe_input.nchw());
            pose_trt.submit(pose_input.nchw());
            try {
                const auto ppe_inference_started = config.telemetry == nullptr
                    ? std::chrono::steady_clock::time_point{} : std::chrono::steady_clock::now();
                const auto ppe_output = ppe_trt.collect();
                if (config.telemetry != nullptr) {
                    config.telemetry->addSample(PerformanceStage::PpeInference,
                        std::chrono::steady_clock::now() - ppe_inference_started);
                }
                const auto ppe_decode_started = config.telemetry == nullptr
                    ? std::chrono::steady_clock::time_point{} : std::chrono::steady_clock::now();
                ppe_detections = decodeDetections(
                    {ppe_output.values, ppe_output.shape},
                    ppe_names.size(), config.ppe_class_confidences, ppe_class_enabled, config.nms_iou,
                    ppe_input.transform,
                    {DecodeLimits{}.max_nms_candidates, config.maximum_detections});
                if (config.telemetry != nullptr) {
                    config.telemetry->addSample(PerformanceStage::PpeDecode,
                        std::chrono::steady_clock::now() - ppe_decode_started);
                }
            } catch (...) {
                try { pose_trt.collect(); } catch (...) {}
                throw;
            }
            const auto pose_inference_started = config.telemetry == nullptr
                ? std::chrono::steady_clock::time_point{} : std::chrono::steady_clock::now();
            const auto pose_output = pose_trt.collect();
            if (config.telemetry != nullptr) {
                config.telemetry->addSample(PerformanceStage::PoseInference,
                    std::chrono::steady_clock::now() - pose_inference_started);
            }
            const auto pose_decode_started = config.telemetry == nullptr
                ? std::chrono::steady_clock::time_point{} : std::chrono::steady_clock::now();
            poses = decodePoses(
                {pose_output.values, pose_output.shape}, pose_class_count,
                static_cast<std::size_t>(keypoint_shape[0]),
                static_cast<std::size_t>(keypoint_shape[1]),
                config.analytics.tracker.low_confidence_threshold,
                config.nms_iou, pose_input.transform,
                {DecodeLimits{}.max_nms_candidates, config.maximum_detections});
            if (config.telemetry != nullptr) {
                config.telemetry->addSample(PerformanceStage::PoseDecode,
                    std::chrono::steady_clock::now() - pose_decode_started);
            }
#endif
        } else {
            try {
                const auto ppe_inference_started = config.telemetry == nullptr
                    ? std::chrono::steady_clock::time_point{} : std::chrono::steady_clock::now();
                const auto ppe_output = ppe_session->infer(ppe_input.nchw());
                if (config.telemetry != nullptr) {
                    config.telemetry->addSample(PerformanceStage::PpeInference,
                        std::chrono::steady_clock::now() - ppe_inference_started);
                }
                const auto ppe_decode_started = config.telemetry == nullptr
                    ? std::chrono::steady_clock::time_point{} : std::chrono::steady_clock::now();
                ppe_detections = decodeDetections(
                    {ppe_output.values, ppe_output.shape},
                    ppe_names.size(), config.ppe_class_confidences, ppe_class_enabled, config.nms_iou,
                    ppe_input.transform,
                    {DecodeLimits{}.max_nms_candidates, config.maximum_detections});
                if (config.telemetry != nullptr) {
                    config.telemetry->addSample(PerformanceStage::PpeDecode,
                        std::chrono::steady_clock::now() - ppe_decode_started);
                }
            } catch (...) {
                if (pose_future.valid()) {
                    try { pose_future.get(); } catch (...) {}
                }
                throw;
            }
            if (hybrid_pose_executor) {
                poses = pose_future.get();
            } else if (pose_session) {
                if (config.pose_requires_person && !personDetectedInPpe(ppe_detections)) {
                    // No person in the frame: skip the whole pose stage.
                    poses.clear();
                } else {
                    PreprocessedFrame pose_input = ppe_input;
                    if (!shared_preprocessing) {
                        const auto pose_preprocess_started = config.telemetry == nullptr
                            ? std::chrono::steady_clock::time_point{} : std::chrono::steady_clock::now();
                        pose_input = pose_preprocessor->process(frame);
                        if (config.telemetry != nullptr) {
                            config.telemetry->addSample(PerformanceStage::PosePreprocess,
                                std::chrono::steady_clock::now() - pose_preprocess_started);
                        }
                    }
                    const auto pose_inference_started = config.telemetry == nullptr
                        ? std::chrono::steady_clock::time_point{} : std::chrono::steady_clock::now();
                    const auto pose_output = pose_session->infer(pose_input.nchw());
                    if (config.telemetry != nullptr) {
                        config.telemetry->addSample(PerformanceStage::PoseInference,
                            std::chrono::steady_clock::now() - pose_inference_started);
                    }
                    const auto pose_decode_started = config.telemetry == nullptr
                        ? std::chrono::steady_clock::time_point{} : std::chrono::steady_clock::now();
                    poses = decodePoses(
                        {pose_output.values, pose_output.shape}, pose_class_count,
                        static_cast<std::size_t>(keypoint_shape[0]),
                        static_cast<std::size_t>(keypoint_shape[1]),
                        config.analytics.tracker.low_confidence_threshold,
                        config.nms_iou, pose_input.transform,
                        {DecodeLimits{}.max_nms_candidates, config.maximum_detections});
                    if (config.telemetry != nullptr) {
                        config.telemetry->addSample(PerformanceStage::PoseDecode,
                            std::chrono::steady_clock::now() - pose_decode_started);
                    }
                }
            }
        }
        const auto analytics_started = config.telemetry == nullptr
            ? std::chrono::steady_clock::time_point{} : std::chrono::steady_clock::now();
        ProcessedFrame result = analytics.process({
            std::string(kContractVersion), std::move(source_id), frame_id,
            monotonic_timestamp_ms, std::move(observed_at), frame.cols, frame.rows,
            std::move(ppe_detections), std::move(poses), ppe_classes,
        });
        if (config.telemetry != nullptr) {
            config.telemetry->addSample(PerformanceStage::Analytics,
                std::chrono::steady_clock::now() - analytics_started);
            config.telemetry->addSample(PerformanceStage::PipelineTotal,
                std::chrono::steady_clock::now() - pipeline_started);
        }
        return result;
    }

    bool personDetectedInPpe(const std::vector<Detection>& detections) const noexcept {
        return std::any_of(detections.begin(), detections.end(), [&](const Detection& detection) {
            return std::find(ppe_classes.person_ids.begin(), ppe_classes.person_ids.end(),
                detection.class_id) != ppe_classes.person_ids.end();
        });
    }

    void cachePpeClassEnabledMask() {
        ppe_class_enabled[1] = 1;
        for (const auto& [item, class_id] : ppe_classes.item_ids) {
            ppe_class_enabled[static_cast<std::size_t>(class_id)] =
                config.ppe_enabled[static_cast<std::size_t>(item)] ? 1 : 0;
        }
    }

    EnginePipelineConfig config;
    EnginePipelineSummary summary;
#ifdef CUAJONE_WITH_TENSORRT
    std::optional<EngineFile> ppe_file;
    std::optional<EngineFile> pose_file;
#endif
    std::map<int, std::string> ppe_names;
    PpeClassMap ppe_classes;
    std::array<std::uint8_t, kPpeOutputLabels.size()> ppe_class_enabled{};
    std::size_t pose_class_count{};
    std::array<int, 2> keypoint_shape{};
    std::unique_ptr<InferenceSession> ppe_session;
    std::unique_ptr<InferenceSession> pose_session;
    std::unique_ptr<LetterboxPreprocessor> ppe_preprocessor;
    std::unique_ptr<LetterboxPreprocessor> pose_preprocessor;
    AnalyticsPipeline analytics;
    std::mutex process_mutex;
    bool hybrid_pose_executor{};
    bool shared_preprocessing{};
    bool tensorrt_gpu_overlap{};
    PoseExecutor pose_executor;
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
    std::scoped_lock lock(impl_->process_mutex);
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
