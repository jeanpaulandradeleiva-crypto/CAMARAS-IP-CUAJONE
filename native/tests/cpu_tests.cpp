#include "cuajone/engine_reader.hpp"
// SPDX-License-Identifier: AGPL-3.0-only

#include "cuajone/cli.hpp"
#include "cuajone/analytics_pipeline.hpp"
#include "cuajone/byte_tracker.hpp"
#include "cuajone/contracts.hpp"
#include "cuajone/fall_analytics.hpp"
#include "cuajone/ppe_analytics.hpp"
#include "cuajone/preprocess.hpp"
#include "cuajone/runtime_execution_plan.hpp"
#include "cuajone/yolo_decode.hpp"

#include <opencv2/core.hpp>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

using namespace cuajone;
using Clock = std::chrono::steady_clock;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void requireNear(float actual, float expected, float tolerance, const std::string& message) {
    if (std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(message + ": expected " + std::to_string(expected)
            + ", got " + std::to_string(actual));
    }
}

template <typename Function>
void requireThrows(Function function, const std::string& message) {
    try {
        function();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(message);
}

class TemporaryFile {
public:
    explicit TemporaryFile(
        const std::vector<unsigned char>& bytes,
        std::string_view extension = ".engine") {
        path_ = std::filesystem::temp_directory_path()
            / ("nexoai_vision_" + std::to_string(++sequence_) + std::string(extension));
        std::ofstream output(path_, std::ios::binary);
        output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }
    ~TemporaryFile() { std::error_code ignored; std::filesystem::remove(path_, ignored); }
    const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
    inline static int sequence_{};
};

void testEngineRawAndMetadataPrefix() {
    TemporaryFile wrong_type({0x24, 0x00, 0x00, 0x00, 0xAA, 0xBB}, ".onnx");
    requireThrows([&] { EngineFile::read(wrong_type.path()); },
        "ONNX artifact was accepted by the TensorRT engine reader");
    TemporaryFile raw({0x24, 0x00, 0x00, 0x00, 0xAA, 0xBB});
    const EngineFile raw_engine = EngineFile::read(raw.path());
    require(!raw_engine.hasMetadataPrefix(), "Raw plan was mistaken for metadata");
    require(raw_engine.plan().size() == 6, "Raw plan length changed");

    const std::string json = R"({"task":"detect","names":{"0":"Person","1":"Hard_hat","2":"Vest"},"imgsz":[640,640],"kpt_shape":[17,3]})";
    std::vector<unsigned char> prefixed{
        static_cast<unsigned char>(json.size() & 0xFFU),
        static_cast<unsigned char>((json.size() >> 8U) & 0xFFU),
        static_cast<unsigned char>((json.size() >> 16U) & 0xFFU),
        static_cast<unsigned char>((json.size() >> 24U) & 0xFFU),
    };
    prefixed.insert(prefixed.end(), json.begin(), json.end());
    prefixed.insert(prefixed.end(), {0x01, 0x02, 0x03});
    TemporaryFile metadata_file(prefixed);
    const EngineFile engine = EngineFile::read(metadata_file.path());
    require(engine.hasMetadataPrefix(), "Metadata prefix was not detected");
    require(engine.plan().size() == 3, "Metadata bytes were not skipped exactly");
    require(engine.metadata().task == "detect", "Task metadata missing");
    require(engine.metadata().names.at(1) == "Hard_hat", "Names metadata missing");
    require(engine.metadata().image_size == std::array{640, 640}, "Image size metadata missing");
    require(engine.metadata().keypoint_shape == std::array{17, 3}, "Keypoint shape metadata missing");

    const std::string invalid = "{invalid";
    std::vector<unsigned char> bad{static_cast<unsigned char>(invalid.size()), 0, 0, 0};
    bad.insert(bad.end(), invalid.begin(), invalid.end());
    bad.push_back(0x01);
    TemporaryFile invalid_file(bad);
    requireThrows([&] { EngineFile::read(invalid_file.path()); }, "Invalid metadata JSON was accepted");

    std::vector<unsigned char> truncated{100, 0, 0, 0, '{', '"', 'x', '"', ':'};
    TemporaryFile truncated_file(truncated);
    requireThrows([&] { EngineFile::read(truncated_file.path()); }, "Truncated metadata prefix was accepted");

    const auto prefixedEngine = [](const std::string& metadata) {
        std::vector<unsigned char> bytes{
            static_cast<unsigned char>(metadata.size() & 0xFFU),
            static_cast<unsigned char>((metadata.size() >> 8U) & 0xFFU),
            static_cast<unsigned char>((metadata.size() >> 16U) & 0xFFU),
            static_cast<unsigned char>((metadata.size() >> 24U) & 0xFFU),
        };
        bytes.insert(bytes.end(), metadata.begin(), metadata.end());
        bytes.push_back(0x01);
        return bytes;
    };
    TemporaryFile surrogate_file(prefixedEngine(R"({"names":{"0":"\uD83D\uDE00"}})"));
    const EngineFile surrogate_engine = EngineFile::read(surrogate_file.path());
    require(surrogate_engine.metadata().names.at(0) == "\xF0\x9F\x98\x80", "Unicode surrogate pair was not combined");

    TemporaryFile lone_surrogate(prefixedEngine(R"({"names":{"0":"\uD83D"}})"));
    requireThrows([&] { EngineFile::read(lone_surrogate.path()); }, "Lone Unicode surrogate was accepted");
    TemporaryFile leading_zero(prefixedEngine(R"({"imgsz":01})"));
    requireThrows([&] { EngineFile::read(leading_zero.path()); }, "JSON leading-zero number was accepted");
    TemporaryFile missing_fraction(prefixedEngine(R"({"imgsz":1.})"));
    requireThrows([&] { EngineFile::read(missing_fraction.path()); }, "JSON number without fraction digits was accepted");
    TemporaryFile missing_exponent(prefixedEngine(R"({"imgsz":1e})"));
    requireThrows([&] { EngineFile::read(missing_exponent.path()); }, "JSON number without exponent digits was accepted");
    TemporaryFile duplicate_ids(prefixedEngine(R"({"names":{"1":"one","01":"also one"}})"));
    requireThrows([&] { EngineFile::read(duplicate_ids.path()); }, "Duplicate numeric class IDs were accepted");
    TemporaryFile oversized_prefix({0x01, 0x00, 0x00, 0x01, '{', '}', 0x01});
    requireThrows([&] { EngineFile::read(oversized_prefix.path()); }, "Recognizable oversized metadata prefix was treated as a raw plan");
}

void testLetterboxMappingAndPacking() {
    requireThrows([] { LetterboxPreprocessor oversized(4097, 1); },
        "Oversized preprocessor dimensions were accepted");
    cv::Mat frame(720, 1280, CV_8UC3, cv::Scalar(10, 20, 30));
    LetterboxPreprocessor preprocessor(640, 640);
    const PreprocessedFrame result = preprocessor.process(frame);
    require(result.nchw.size() == 3U * 640U * 640U, "Unexpected NCHW length");
    require(result.transform.padding_left == 0, "Unexpected horizontal padding");
    require(result.transform.padding_top == 140, "Unexpected vertical padding");
    const Box restored = result.transform.restore(Box{0.0F, 140.0F, 640.0F, 500.0F});
    requireNear(restored.x2, 1280.0F, 0.01F, "Box x mapping failed");
    requireNear(restored.y2, 720.0F, 0.01F, "Box y mapping failed");
    const std::size_t image_offset = 140U * 640U;
    requireNear(result.nchw[image_offset], 30.0F / 255.0F, 0.0001F, "BGR to RGB failed");
}

void testDetectSchemaDecodeAndNms() {
    const LetterboxTransform transform{640, 640, 640, 640, 1.0F, 1.0F, 0, 0};
    // [1, channels=7, predictions=2], channel-major.
    const std::vector<float> values{
        100, 400,
        100, 400,
        40, 40,
        60, 60,
        0.90F, 0.10F,
        0.05F, 0.20F,
        0.01F, 0.30F,
    };
    const std::array<std::int64_t, 3> shape{1, 7, 2};
    const TensorView tensor{values, shape};
    const auto schema = validateDetectSchema(tensor.shape, 3);
    require(schema.layout == TensorLayout::ChannelsFirst, "Detect layout was not recognized");
    const auto decoded = decodeDetections(tensor, 3, 0.5F, 0.45F, transform);
    require(decoded.size() == 1 && decoded[0].class_id == 0, "Detect decode confidence failed");
    requireNear(decoded[0].box.x1, 80.0F, 0.01F, "Detect box decode failed");

    const std::vector<float> objectness_values{100, 100, 40, 60, 0.8F, 0.75F, 0.1F, 0.1F};
    const std::array<std::int64_t, 3> objectness_shape{1, 1, 8};
    const TensorView objectness_tensor{objectness_values, objectness_shape};
    const auto objectness_decoded = decodeDetections(objectness_tensor, 3, 0.5F, 0.45F, transform);
    require(objectness_decoded.size() == 1, "Objectness detect schema was not decoded");
    requireNear(objectness_decoded[0].confidence, 0.6F, 0.001F, "Objectness was not multiplied");
    requireThrows(
        [] { validateDetectSchema(std::array<std::int64_t, 3>{1, 6, 100}, 3); },
        "Unsupported detect schema was accepted");

    std::vector<Detection> candidates{
        {{0, 0, 100, 100}, 0.9F, 0},
        {{5, 5, 95, 95}, 0.8F, 0},
        {{5, 5, 95, 95}, 0.7F, 1},
    };
    const auto kept = classAwareNms(std::move(candidates), 0.45F);
    require(kept.size() == 2, "Class-aware NMS suppressed the wrong boxes");

    const std::vector<float> edge_values{
        10, 30,
        50, 50,
        60, 60,
        100, 100,
        0.9F, 0.8F,
    };
    const std::array<std::int64_t, 3> edge_shape{1, 5, 2};
    const auto edge_detections = decodeDetections(
        {edge_values, edge_shape}, 1, 0.1F, 0.6F,
        LetterboxTransform{100, 100, 100, 100, 1.0F, 1.0F, 0, 0});
    require(edge_detections.size() == 2, "NMS ran after clipping instead of in model coordinates");

    auto nonfinite = objectness_values;
    nonfinite[4] = std::numeric_limits<float>::quiet_NaN();
    require(decodeDetections({nonfinite, objectness_shape}, 3, 0.1F, 0.45F, transform).empty(),
        "Non-finite objectness was accepted");
    const auto limited = decodeDetections(
        {edge_values, edge_shape}, 1, 0.1F, 1.0F, transform, {2, 1});
    require(limited.size() == 1 && limited[0].confidence == 0.9F,
        "Final max detection limit did not preserve the best candidate");
}

void testPoseSchemaAndDecode() {
    const OnnxPoseContract contract = validateOnnxPoseContract(1, {17, 3});
    require(contract.class_count == 1 && contract.keypoint_shape == std::array{17, 3},
        "ONNX pose contract did not preserve configured decode dimensions");
    requireThrows([] { validateOnnxPoseContract(2, {17, 3}); },
        "ONNX pose contract accepted multiple classes");
    requireThrows([] { validateOnnxPoseContract(1, {17, 2}); },
        "ONNX pose contract accepted incomplete keypoints");
    requireThrows([] { validateOnnxPoseContract(1, {0, 3}); },
        "ONNX pose contract accepted zero keypoints");

    const LetterboxTransform transform{640, 640, 640, 640, 1.0F, 1.0F, 0, 0};
    // [1, predictions=1, channels=11]: xywh + one class + 2*(x,y,confidence).
    const std::vector<float> values{
        100, 120, 40, 80, 0.8F,
        90, 100, 0.7F,
        110, 140, 0.6F,
    };
    const std::array<std::int64_t, 3> shape{1, 1, 11};
    const TensorView tensor{values, shape};
    const auto schema = validatePoseSchema(tensor.shape, 1, 2, 3);
    require(schema.layout == TensorLayout::PredictionsFirst, "Pose layout was not recognized");
    const auto poses = decodePoses(tensor, 1, 2, 3, 0.5F, 0.45F, transform);
    require(poses.size() == 1 && poses[0].keypoints.size() == 2, "Pose decode failed");
    requireNear(poses[0].keypoints[1].confidence, 0.6F, 0.001F, "Keypoint confidence lost");
    requireThrows(
        [] { validatePoseSchema(std::array<std::int64_t, 3>{1, 56, 56}, 1, 17, 3); },
        "Ambiguous pose schema was accepted");

    auto nonfinite = values;
    nonfinite[5] = std::numeric_limits<float>::infinity();
    require(decodePoses({nonfinite, shape}, 1, 2, 3, 0.5F, 0.45F, transform).empty(),
        "Non-finite pose keypoint coordinate was accepted");

    constexpr std::size_t keypoint_count = 17;
    constexpr std::size_t end_to_end_channels = 6 + keypoint_count * 3;
    std::vector<float> end_to_end_values(300 * end_to_end_channels, 0.0F);
    end_to_end_values[0] = 80.0F;
    end_to_end_values[1] = 60.0F;
    end_to_end_values[2] = 240.0F;
    end_to_end_values[3] = 500.0F;
    end_to_end_values[4] = 0.90F;
    end_to_end_values[5] = 0.0F;
    for (std::size_t keypoint = 0; keypoint < keypoint_count; ++keypoint) {
        const std::size_t offset = 6 + keypoint * 3;
        end_to_end_values[offset] = 100.0F + static_cast<float>(keypoint);
        end_to_end_values[offset + 1] = 120.0F + static_cast<float>(keypoint);
        end_to_end_values[offset + 2] = 0.80F;
    }
    const std::array<std::int64_t, 3> end_to_end_shape{1, 300, 57};
    const auto end_to_end_schema = validatePoseSchema(
        end_to_end_shape, 1, keypoint_count, 3);
    require(end_to_end_schema.format == YoloOutputFormat::PoseEndToEnd
            && end_to_end_schema.layout == TensorLayout::PredictionsFirst,
        "Approved [1,300,57] pose output was not recognized as end-to-end");
    const auto end_to_end = decodePoses(
        {end_to_end_values, end_to_end_shape}, 1, keypoint_count, 3,
        0.35F, 0.45F, transform);
    require(end_to_end.size() == 1 && end_to_end.front().class_id == 0
            && end_to_end.front().keypoints.size() == keypoint_count,
        "End-to-end pose row did not produce the person detection");
    requireNear(end_to_end.front().confidence, 0.90F, 0.0001F,
        "End-to-end confidence was multiplied by class_id");
    requireNear(end_to_end.front().box.x1, 80.0F, 0.01F,
        "End-to-end xyxy box was decoded as xywh");

    auto invalid_class = end_to_end_values;
    invalid_class[5] = 0.5F;
    require(decodePoses(
                {invalid_class, end_to_end_shape}, 1, keypoint_count, 3,
                0.35F, 0.45F, transform).empty(),
        "Fractional end-to-end class ID was accepted");
}

void testCliUrlsAndInvariantDefense() {
    require(isPersonClassLabel("Person") && isPersonClassLabel(" persona ")
            && !isPersonClassLabel("worker"),
        "Pose person/persona metadata normalization contract failed");
    validateRtspSource("rtsp://user:password@[2001:db8::1]:554/live");
    validateRtspSource("rtsps://camera.example/live");
    require(defaultSourceLabel("rtsp://user:password@[2001:db8::1]:554/live")
            == "rtsp-2001:db8::1",
        "RTSP IPv6 source label was not credential-free");
    require(redactSource("rtsps://user:password@camera.example/live")
            == "rtsps://***@camera.example/live",
        "RTSPS credentials were not redacted");
    requireThrows([] { validateRtspSource("rtsp://"); }, "Empty RTSP authority was accepted");
    requireThrows([] { validateRtspSource("rtsp://user:password@"); }, "Empty RTSP host was accepted");
    requireThrows([] { validateRtspSource("rtsp://2001:db8::1/live"); }, "Unbracketed IPv6 host was accepted");

    const auto parse = [](std::vector<std::string> arguments) {
        std::vector<char*> argv;
        argv.reserve(arguments.size());
        for (auto& argument : arguments) argv.push_back(argument.data());
        return parseCommandLine(static_cast<int>(argv.size()), argv.data());
    };
    const std::vector<std::string> base{
        "NexoAIVision", "--source", "video.mp4", "--ppe-engine", "ppe.engine",
        "--pose-engine", "pose.engine", "--output", "out",
    };
    auto valid = base;
    valid.insert(valid.end(), {"--max-det", "42", "--rtsp-transport", "udp",
                               "--capture-open-timeout-ms", "0", "--device", "2"});
    const RuntimeConfig parsed = parse(valid);
    require(parsed.max_det == 42 && parsed.rtsp_transport == RtspTransport::Udp
            && parsed.capture_open_timeout.count() == 0 && parsed.device == 2,
        "CLI did not preserve max-det, transport, zero timeout, or device");
    auto byte_track = base;
    byte_track.insert(byte_track.end(), {
        "--tracker-high-threshold", "0.4", "--tracker-match-threshold", "0.7",
        "--tracker-max-age", "45", "--tracker-max-tracks", "64",
        "--tracker-frame-rate", "25",
    });
    const RuntimeConfig parsed_byte_track = parse(byte_track);
    require(parsed_byte_track.tracker_high_threshold == 0.4F
            && parsed_byte_track.tracker_match_threshold == 0.7F
            && parsed_byte_track.tracker_max_age == 45
            && parsed_byte_track.tracker_max_tracks == 64
            && parsed_byte_track.tracker_frame_rate == 25,
        "CLI did not preserve ByteTrack configuration");
    auto legacy_tracker_iou = base;
    legacy_tracker_iou.insert(legacy_tracker_iou.end(), {"--tracker-iou", "0.3"});
    requireNear(parse(legacy_tracker_iou).tracker_match_threshold, 0.7F, 0.0001F,
        "Legacy tracker IoU alias was not converted to ByteTrack match cost");
    auto invalid_confidence = base;
    invalid_confidence.insert(invalid_confidence.end(), {"--pose-conf", "1.1"});
    requireThrows([&] { parse(invalid_confidence); }, "CLI accepted confidence above one");
    auto invalid_cooldown = base;
    invalid_cooldown.insert(invalid_cooldown.end(), {"--fall-cooldown", "-1"});
    requireThrows([&] { parse(invalid_cooldown); }, "CLI accepted a negative cooldown");
    auto invalid_capacity = base;
    invalid_capacity.insert(invalid_capacity.end(), {"--tracker-max-tracks", "0"});
    requireThrows([&] { parse(invalid_capacity); }, "CLI accepted zero tracker capacity");
    const std::vector<std::string> ppe_only{
        "NexoAIVision", "--mode", "ppe-only", "--source", "video.mp4",
        "--ppe-engine", "ppe.engine", "--output", "out",
    };
    require(parse(ppe_only).analytics_mode == AnalyticsMode::PpeOnly,
        "PPE-only unexpectedly required a pose engine");
    auto invalid_mode = ppe_only;
    invalid_mode[2] = "unknown";
    requireThrows([&] { parse(invalid_mode); }, "Unknown analytics mode was accepted");

    auto cpu = ppe_only;
    cpu.erase(cpu.begin() + 5, cpu.begin() + 7);
    cpu.insert(cpu.end(), {"--compute", "cpu", "--ppe-onnx", "ppe.onnx", "--ppe-labels", "person,helmet,vest"});
    const RuntimeConfig parsed_cpu = parse(cpu);
    require(parsed_cpu.compute_backend == ComputeBackend::Cpu && parsed_cpu.compute_explicit
            && parsed_cpu.ppe_onnx == "ppe.onnx",
        "CLI did not preserve explicit CPU compute selection");
    auto invalid_compute = ppe_only;
    invalid_compute.insert(invalid_compute.end(), {"--compute", "gpu"});
    requireThrows([&] { parse(invalid_compute); }, "CLI accepted an unknown compute backend");

    requireThrows(
        [] { ByteTracker({0.35F, 0.10F, 0.80F, 0, 1, 30}); },
        "Zero tracker age was accepted by the constructor");
    requireThrows(
        [] { PpeAnalyzer({1, 1, 0.5F, std::chrono::seconds(-1), std::chrono::seconds(1)}); },
        "Negative PPE cooldown was accepted by the constructor");
    requireThrows(
        [] { FallAnalyzer({0, 1, std::chrono::seconds(1), std::chrono::seconds(1), 1.0F, 45.0F, 0.1F, 0.5F}); },
        "Zero fall confirmation count was accepted by the constructor");
}

void testComputeSelectionAndProbeContract() {
    require(!isTensorRtCompatibleComputeCapability(7, 0)
            && isTensorRtCompatibleComputeCapability(7, 5)
            && isTensorRtCompatibleComputeCapability(8, 6),
        "TensorRT 11 compute-capability floor changed");
    require(parseComputeBackend("auto") == ComputeBackend::Auto
            && parseComputeBackend("cuda") == ComputeBackend::Cuda
            && parseComputeBackend("cpu") == ComputeBackend::Cpu,
        "Compute backend parser changed");
    require(selectComputeBackend(ComputeBackend::Auto, {
                HardwareProbeStatus::NoNvidiaAdapter, false, false, false, true,
            }).provider == InferenceProvider::OnnxRuntimeCpu,
        "Auto did not route ONNX models to the CPU provider without CUDA readiness");
    require(selectComputeBackend(ComputeBackend::Auto, {
                HardwareProbeStatus::CudaReady, false, true, false, true,
            }).provider == InferenceProvider::OnnxRuntimeCuda,
        "Auto did not route supported ONNX models to the CUDA execution provider");
    require(selectComputeBackend(ComputeBackend::Auto, {
                HardwareProbeStatus::CudaReady, true, true, true, true,
            }).provider == InferenceProvider::TensorRt,
        "Auto did not preserve TensorRT routing precedence over ONNX CUDA");
    const std::array statuses{
        HardwareProbeStatus::NoNvidiaAdapter,
        HardwareProbeStatus::DriverUnavailable,
        HardwareProbeStatus::DriverTooOld,
        HardwareProbeStatus::CudaReady,
        HardwareProbeStatus::ProbeError,
    };
    for (const ComputeBackend requested : {
             ComputeBackend::Auto, ComputeBackend::Cuda, ComputeBackend::Cpu}) {
        for (const HardwareProbeStatus status : statuses) {
            for (const bool tensor_rt_runtime_compiled : {false, true}) {
                for (const bool onnx_cuda_compiled : {false, true}) {
                    for (const bool tensor_rt_models : {false, true}) {
                        for (const bool onnx_models : {false, true}) {
                            std::optional<ComputeBackend> expected;
                            std::optional<InferenceProvider> expected_provider;
                            if (status == HardwareProbeStatus::CudaReady) {
                                if (tensor_rt_runtime_compiled && tensor_rt_models) {
                                    expected_provider = InferenceProvider::TensorRt;
                                } else if (onnx_cuda_compiled && onnx_models) {
                                    expected_provider = InferenceProvider::OnnxRuntimeCuda;
                                }
                            }
                            if (requested == ComputeBackend::Cpu) {
                                if (onnx_models) {
                                    expected = ComputeBackend::Cpu;
                                    expected_provider = InferenceProvider::OnnxRuntimeCpu;
                                }
                            } else if (requested == ComputeBackend::Cuda) {
                                if (expected_provider) {
                                    expected = ComputeBackend::Cuda;
                                }
                            } else if (expected_provider) {
                                expected = ComputeBackend::Cuda;
                            } else if (onnx_models) {
                                expected = ComputeBackend::Cpu;
                                expected_provider = InferenceProvider::OnnxRuntimeCpu;
                            }
                            std::optional<ComputeSelection> actual;
                            try {
                                actual = selectComputeBackend(requested, {
                                status, tensor_rt_runtime_compiled, onnx_cuda_compiled,
                                    tensor_rt_models, onnx_models,
                                });
                            } catch (const std::exception&) {
                            }
                            require(actual.has_value() == expected.has_value(),
                                "Compute matrix availability result changed");
                            if (expected) require(actual->backend == *expected,
                                "Compute matrix selected the wrong available backend");
                            if (expected && expected_provider) require(actual->provider == *expected_provider,
                                "Compute matrix selected the wrong inference provider");
                        }
                    }
                }
            }
        }
    }

    const std::array devices{
        CudaDeviceInfo{0, "Legacy", 7, 0},
        CudaDeviceInfo{1, "Compatible A", 8, 6},
        CudaDeviceInfo{2, "Compatible B", 7, 5},
    };
    require(selectCompatibleCudaDevice(devices, std::nullopt) == 1,
        "Automatic CUDA device selection did not choose the first compatible device");
    require(selectCompatibleCudaDevice(devices, 2) == 2,
        "Explicit compatible CUDA device was not preserved");
    requireThrows([&] { selectCompatibleCudaDevice(devices, 0); },
        "Explicit incompatible CUDA device was accepted");
    requireThrows([&] { selectCompatibleCudaDevice(devices, 3); },
        "Unavailable CUDA device index was accepted");
    const std::array incompatible{CudaDeviceInfo{0, "Legacy", 7, 0}};
    requireThrows([&] { selectCompatibleCudaDevice(incompatible, std::nullopt); },
        "Automatic selection accepted a device below SM 7.5");
    require(kMinimumCudaDriverApiVersion == 12090, "CUDA Driver API floor changed");

    HardwareProbeResult synthetic;
    synthetic.status = HardwareProbeStatus::CudaReady;
    synthetic.driver_version = 12090;
    synthetic.adapters.push_back({"Synthetic NVIDIA", 1234, 4096});
    synthetic.cuda_devices.push_back({0, "Synthetic CUDA", 8, 6});
    synthetic.detail = "synthetic";
    const std::string json = hardwareProbeJson(synthetic);
    require(json.find("\"schema_version\":2") != std::string::npos
            && json.find("\"status\":\"cuda_ready\"") != std::string::npos
            && json.find("\"cuda_ready\":true") != std::string::npos
            && json.find("\"minimum_driver_version\":12090") != std::string::npos
            && json.find("\"cuda_driver_api\":\"12.9\"") != std::string::npos
            && json.find("\"device_index\":0") != std::string::npos
            && json.find("\"driver_was_loaded\":false") != std::string::npos,
        "Hardware probe JSON contract changed");
    require(hardwareProbeSummary(synthetic).find("CUDA Driver API 12.9") != std::string::npos
            && hardwareProbeSummary(synthetic).find("Synthetic CUDA") != std::string::npos,
        "Hardware probe summary no longer reports CUDA version and GPU name");
    require(hardwareProbeExitCode(HardwareProbeStatus::CudaReady) == 0
            && hardwareProbeExitCode(HardwareProbeStatus::NoNvidiaAdapter) == 10
            && hardwareProbeExitCode(HardwareProbeStatus::DriverUnavailable) == 11
            && hardwareProbeExitCode(HardwareProbeStatus::ProbeError) == 12
            && hardwareProbeExitCode(HardwareProbeStatus::DriverTooOld) == 13,
        "Hardware probe exit-code contract changed");
}

void testRuntimeExecutionPlanning() {
    const ModelArtifactAvailability all_models{true, true, true, true};
    const ModelArtifactAvailability ppe_only_onnx{false, false, true, false};
    struct PlanCase {
        std::string name;
        RuntimeExecutionPlanningInput input;
        std::optional<ComputeBackend> backend;
        std::optional<InferenceProvider> provider;
        std::optional<ComputeBackend> fallback;
        bool pose_required{};
        std::string error;
    };
    const std::vector<PlanCase> cases{
        {
            "Auto ONNX CUDA falls back to CPU after preflight",
            {ComputeBackend::Auto, true, std::nullopt, AnalyticsMode::PpeFall,
             {HardwareProbeStatus::CudaReady, false, true}, all_models},
            ComputeBackend::Cuda, InferenceProvider::OnnxRuntimeCuda,
            ComputeBackend::Cpu, true, {},
        },
        {
            "CPU selects ONNX CPU without a CUDA fallback",
            {ComputeBackend::Cpu, true, std::nullopt, AnalyticsMode::PpeFall,
             {HardwareProbeStatus::NoNvidiaAdapter, false, false}, all_models},
            ComputeBackend::Cpu, InferenceProvider::OnnxRuntimeCpu,
            std::nullopt, true, {},
        },
        {
            "CUDA selects ONNX CUDA without a CPU fallback",
            {ComputeBackend::Cuda, true, std::nullopt, AnalyticsMode::PpeFall,
             {HardwareProbeStatus::CudaReady, false, true}, all_models},
            ComputeBackend::Cuda, InferenceProvider::OnnxRuntimeCuda,
            std::nullopt, true, {},
        },
        {
            "Auto prefers TensorRT over ONNX CUDA",
            {ComputeBackend::Auto, true, std::nullopt, AnalyticsMode::PpeFall,
             {HardwareProbeStatus::CudaReady, true, true}, all_models},
            ComputeBackend::Cuda, InferenceProvider::TensorRt,
            ComputeBackend::Cpu, true, {},
        },
        {
            "PPE-only ONNX does not require a pose model",
            {ComputeBackend::Cpu, true, std::nullopt, AnalyticsMode::PpeOnly,
             {HardwareProbeStatus::NoNvidiaAdapter, false, false}, ppe_only_onnx},
            ComputeBackend::Cpu, InferenceProvider::OnnxRuntimeCpu,
            std::nullopt, false, {},
        },
        {
            "CPU plan rejects missing ONNX models",
            {ComputeBackend::Cpu, true, std::nullopt, AnalyticsMode::PpeFall,
             {HardwareProbeStatus::NoNvidiaAdapter, false, false}, {}},
            std::nullopt, std::nullopt, std::nullopt, true,
            "CPU mode requires compatible PPE and pose ONNX models",
        },
        {
            "CUDA plan rejects an unavailable runtime",
            {ComputeBackend::Cuda, true, std::nullopt, AnalyticsMode::PpeFall,
             {HardwareProbeStatus::CudaReady, false, false}, all_models},
            std::nullopt, std::nullopt, std::nullopt, true,
            "CUDA mode is unavailable in this CPU-only build",
        },
        {
            "Auto plan rejects absent model paths",
            {ComputeBackend::Auto, true, std::nullopt, AnalyticsMode::PpeFall,
             {HardwareProbeStatus::NoNvidiaAdapter, false, false}, {}},
            std::nullopt, std::nullopt, std::nullopt, true,
            "Auto mode found neither a ready TensorRT path nor compatible ONNX models",
        },
    };
    for (const auto& test : cases) {
        try {
            const RuntimeExecutionPlan plan = planRuntimeExecution(test.input);
            require(test.error.empty(), test.name + " did not reject its invalid plan");
            require(plan.selection.backend == *test.backend, test.name + " selected the wrong backend");
            require(plan.selection.provider == *test.provider, test.name + " selected the wrong provider");
            require(plan.model_requirements.pose_required == test.pose_required,
                test.name + " resolved pose requirements incorrectly");
            require(plan.preflight_failure_fallback.has_value() == test.fallback.has_value(),
                test.name + " resolved the wrong fallback policy");
            if (test.fallback) {
                require(plan.preflight_failure_fallback->backend == *test.fallback,
                    test.name + " selected the wrong fallback backend");
            }
        } catch (const std::exception& error) {
            require(!test.error.empty(), test.name + " unexpectedly failed: " + error.what());
            require(std::string(error.what()).find(test.error) != std::string::npos,
                test.name + " returned the wrong error: " + error.what());
        }
    }
}

void testByteTrackLifecycleAndLowConfidenceAssociation() {
    ByteTracker tracker({0.35F, 0.10F, 0.80F, 2, 4, 30});
    const std::array<TrackingDetection, 1> first{{{{0, 0, 100, 100}, 0.90F}}};
    const int id = tracker.update(first)[0];
    require(id > 0, "ByteTrack did not activate a high-confidence detection");
    const std::array<TrackingDetection, 1> moved{{{{5, 0, 105, 100}, 0.90F}}};
    require(tracker.update(moved)[0] == id, "ByteTrack did not preserve ID");
    tracker.update({});
    const std::array<TrackingDetection, 1> occluded_return{{{{10, 0, 110, 100}, 0.90F}}};
    require(tracker.update(occluded_return)[0] == id,
        "ByteTrack did not reacquire a briefly occluded track");
    const std::array<TrackingDetection, 1> low_confidence{{{{12, 0, 112, 100}, 0.20F}}};
    require(tracker.update(low_confidence)[0] == id,
        "ByteTrack did not use a low-confidence detection in second-stage association");
    tracker.update({});
    tracker.update({});
    require(tracker.activeTrackCount() == 1, "ByteTrack expired too early");
    tracker.update({});
    require(tracker.activeTrackCount() == 0, "ByteTrack did not expire after the lost buffer");
}

void testByteTrackCapacityResetEmptyFramesAndTies() {
    ByteTracker empty_tracker({0.35F, 0.10F, 0.80F, 2, 2, 30});
    require(empty_tracker.update({}).empty() && empty_tracker.activeTrackCount() == 0,
        "Empty ByteTrack frame created tracks");
    ByteTracker tracker({0.35F, 0.10F, 0.80F, 2, 2, 30});
    const std::array<TrackingDetection, 3> detections{{
        {{0, 0, 100, 100}, 0.90F},
        {{200, 0, 300, 100}, 0.90F},
        {{400, 0, 500, 100}, 0.90F},
    }};
    const auto ids = tracker.update(detections);
    const auto assigned = std::count_if(ids.begin(), ids.end(), [](int id) { return id > 0; });
    require(assigned == 2 && tracker.activeTrackCount() == 2,
        "ByteTrack capacity mismatch: assigned=" + std::to_string(assigned)
            + ", retained=" + std::to_string(tracker.activeTrackCount()));
    tracker.reset();
    require(tracker.activeTrackCount() == 0,
        "ByteTrack reset did not clear retained state");
    const int post_reset_id = tracker.update(std::span(detections).first<1>())[0];
    require(post_reset_id > 0
            && std::find(ids.begin(), ids.end(), post_reset_id) == ids.end(),
        "ByteTrack reset reused a process-wide identity");

    ByteTracker tied({0.35F, 0.10F, 0.80F, 2, 2, 30});
    const std::array<TrackingDetection, 2> identical{{
        {{10, 10, 110, 110}, 0.90F},
        {{10, 10, 110, 110}, 0.90F},
    }};
    const auto tied_ids = tied.update(identical);
    require(tied_ids[0] > 0 && tied_ids[1] > 0 && tied_ids[0] != tied_ids[1],
        "ByteTrack tie handling silently assigned one ID twice");

    ByteTracker first_camera;
    ByteTracker second_camera;
    const std::array<TrackingDetection, 1> person{{{{0, 0, 100, 100}, 0.90F}}};
    int first_camera_id{};
    int second_camera_id{};
    std::jthread first_thread([&] { first_camera_id = first_camera.update(person)[0]; });
    std::jthread second_thread([&] { second_camera_id = second_camera.update(person)[0]; });
    first_thread.join();
    second_thread.join();
    require(first_camera_id > 0 && second_camera_id > 0
            && first_camera_id != second_camera_id,
        "Constructing another camera tracker reset the process-wide identity counter");

    ByteTracker long_running({0.35F, 0.10F, 0.80F, 1, 1, 30});
    for (int cycle = 0; cycle < 64; ++cycle) {
        const float offset = static_cast<float>(cycle * 200);
        const std::array<TrackingDetection, 1> detection{{{
            {offset, 0, offset + 100, 100}, 0.90F,
        }}};
        long_running.update(detection);
        require(long_running.update(detection)[0] > 0
                && long_running.activeTrackCount() == 1,
            "Long-running ByteTrack cycle did not activate within capacity");
        long_running.update({});
        long_running.update({});
        require(long_running.activeTrackCount() == 0,
            "Removed ByteTrack state grew across long-running cycles");
    }
}

void testPpeAssociationVotingAndCooldown() {
    const auto classes = resolvePpeClasses({{0, "Person"}, {1, "Hard hat"}, {2, "Vest"}});
    const TrackedPerson person{7, {100, 50, 300, 450}, 0.9F, {}, true};
    const std::array<Detection, 2> items{{
        {{150, 40, 220, 130}, 0.8F, 1},
        {{140, 160, 260, 330}, 0.9F, 2},
    }};
    const auto associations = associatePpe(std::span(&person, 1), items, classes);
    require(associations.at(7).helmet && associations.at(7).vest, "PPE association failed");

    PpeAnalyzer analyzer({4, 3, 0.5F, std::chrono::seconds(10), std::chrono::seconds(5)});
    const PpeAssociation missing{};
    const auto start = Clock::time_point{} + std::chrono::seconds(100);
    require(!analyzer.update(7, missing, true, start), "PPE voted before minimum samples");
    require(!analyzer.update(7, missing, true, start + std::chrono::seconds(1)), "PPE voted early");
    require(analyzer.update(7, missing, true, start + std::chrono::seconds(2)).has_value(), "PPE violation was not emitted");
    require(!analyzer.update(7, missing, true, start + std::chrono::seconds(3)), "PPE cooldown failed");
    require(analyzer.update(7, missing, true, start + std::chrono::seconds(12)).has_value(), "PPE cooldown did not reopen");
}

std::vector<Keypoint> horizontalPose() {
    std::vector<Keypoint> points(17, {0, 0, 0.0F});
    points[5] = {100, 300, 0.9F};
    points[6] = {100, 320, 0.9F};
    points[11] = {250, 300, 0.9F};
    points[12] = {250, 320, 0.9F};
    points[0] = {50, 300, 0.9F};
    return points;
}

std::vector<Keypoint> uprightPose() {
    std::vector<Keypoint> points(17, {0, 0, 0.0F});
    points[5] = {180, 180, 0.9F};
    points[6] = {220, 180, 0.9F};
    points[11] = {180, 360, 0.9F};
    points[12] = {220, 360, 0.9F};
    points[0] = {200, 100, 0.9F};
    return points;
}

void testFallConfirmationRecoveryAndCooldown() {
    FallAnalyzer analyzer({
        3, 2, std::chrono::seconds(20), std::chrono::seconds(5),
        1.05F, 55.0F, 0.12F, 0.65F,
    });
    const auto pose = horizontalPose();
    const auto upright_pose = uprightPose();
    PoseDetection weak_pose;
    weak_pose.box = {100, 100, 300, 600};
    weak_pose.confidence = 0.4F;
    weak_pose.keypoints = upright_pose;
    const std::array<Box, 1> ppe_anchor{{{110, 110, 290, 590}}};
    require(isValidPosePerson(weak_pose, ppe_anchor, 0.35F, 0.45F), "PPE anchor did not validate pose person");
    const Box fallen{100, 500, 400, 650};
    const auto start = Clock::time_point{} + std::chrono::seconds(100);
    require(!analyzer.update(9, fallen, pose, 720, start).confirmed_now, "Fall confirmed early");
    require(!analyzer.update(9, fallen, pose, 720, start + std::chrono::seconds(1)).confirmed_now, "Fall confirmed early");
    require(analyzer.update(9, fallen, pose, 720, start + std::chrono::seconds(2)).confirmed_now, "Fall was not confirmed");

    const Box upright{150, 100, 260, 600};
    analyzer.update(9, upright, upright_pose, 720, start + std::chrono::seconds(3));
    const auto recovered = analyzer.update(9, upright, upright_pose, 720, start + std::chrono::seconds(4));
    require(!recovered.active, "Fall state did not recover");
    require(!analyzer.update(9, fallen, pose, 720, start + std::chrono::seconds(5)).confirmed_now, "Fall cooldown failed");
    analyzer.update(9, fallen, pose, 720, start + std::chrono::seconds(6));
    require(!analyzer.update(9, fallen, pose, 720, start + std::chrono::seconds(7)).confirmed_now, "Fall cooldown failed");
    analyzer.update(9, upright, upright_pose, 720, start + std::chrono::seconds(8));
    analyzer.update(9, upright, upright_pose, 720, start + std::chrono::seconds(9));
    analyzer.update(9, fallen, pose, 720, start + std::chrono::seconds(23));
    analyzer.update(9, fallen, pose, 720, start + std::chrono::seconds(24));
    require(analyzer.update(9, fallen, pose, 720, start + std::chrono::seconds(25)).confirmed_now, "Fall cooldown did not reopen");
}

ObservationFrame syntheticPpeFrame(std::uint64_t frame_id, std::int64_t timestamp_ms) {
    return {
        std::string(kContractVersion),
        "SYNTHETIC_QA_01",
        frame_id,
        timestamp_ms,
        frame_id == 1 ? "2026-01-01T00:00:00.100Z" : "2026-01-01T00:00:00.200Z",
        640,
        720,
        {{{100, 100, 300, 500}, 0.9F, 0}},
        {},
        {{0}, {1}, {2}},
    };
}

void testPoseFilteringPreservesTrackerIdentity() {
    AnalyticsPipelineConfig config;
    config.mode = AnalyticsMode::PpeFall;
    config.ppe = {1, 1, 0.5F, std::chrono::seconds(60), std::chrono::seconds(5)};
    AnalyticsPipeline pipeline(config);

    PoseDetection invalid;
    invalid.box = {10, 10, 80, 180};
    invalid.confidence = 0.9F;
    PoseDetection valid;
    valid.box = {100, 100, 300, 600};
    valid.confidence = 0.9F;
    valid.keypoints = uprightPose();
    ObservationFrame frame{
        std::string(kContractVersion),
        "POSE_ORDER_QA",
        1,
        100,
        "2026-01-01T00:00:00.100Z",
        640,
        720,
        {{{100, 100, 300, 600}, 0.9F, 0}},
        {invalid, valid},
        {{0}, {1}, {2}},
    };

    const auto result = pipeline.process(frame);
    require(result.canonical.people.size() == 1, "Invalid pose was not filtered after tracking");
    const int valid_track_id = result.canonical.people.front().track_id;
    require(valid_track_id > 0,
        "Filtering invalid pose removed the valid ByteTrack identity");
    require(result.canonical.events.size() == 1
            && result.canonical.events.front().track_id == valid_track_id
            && result.canonical.events.front().id
                == "evt-POSE_ORDER_QA-1-" + std::to_string(valid_track_id) + "-0",
        "Pose filtering changed the pre-refactor event identity");
}

void testCanonicalContractsAndDeterministicPipeline() {
    AnalyticsPipelineConfig config;
    config.mode = AnalyticsMode::PpeOnly;
    config.ppe = {2, 2, 0.5F, std::chrono::seconds(60), std::chrono::seconds(5)};
    AnalyticsPipeline pipeline(config);
    require(pipeline.contractVersion() == "1.0.0", "Pipeline contract version changed");
    require(runtimeDefaultsJson().find("\"profile\":\"byte-track-eigen\"") != std::string::npos,
        "Canonical defaults lost the ByteTrack-Eigen profile");

    const auto first = pipeline.process(syntheticPpeFrame(1, 100));
    require(first.canonical.people.size() == 1 && first.canonical.events.empty(),
        "PPE pipeline emitted before the minimum sample count");
    const auto second = pipeline.process(syntheticPpeFrame(2, 200));
    require(second.canonical.events.size() == 1, "PPE pipeline did not emit deterministically");
    const std::string expected = canonicalJson(second.canonical);
    require(expected.find("INCUMPLIMIENTO_EPP") == std::string::npos,
        "Frame result leaked the legacy event name instead of canonical references");
    require(expected.find("evt-SYNTHETIC_QA_01-2-"
                + std::to_string(second.canonical.people.front().track_id) + "-0")
            != std::string::npos,
        "Canonical event ID changed");
    const std::string event_json = canonicalJson(second.canonical.events.front());
    require(event_json.find("\"specversion\":\"1.0\"") != std::string::npos,
        "CloudEvents version is missing");
    require(event_json.find("rtsp://") == std::string::npos,
        "Canonical event exposed an RTSP URL");
    requireThrows([&] { pipeline.process(syntheticPpeFrame(2, 300)); },
        "Duplicate frame_id was accepted");

    const int first_sequence_id = second.canonical.people.front().track_id;
    pipeline.reset();
    pipeline.process(syntheticPpeFrame(1, 100));
    const auto repeated = pipeline.process(syntheticPpeFrame(2, 200));
    require(repeated.canonical.people.size() == 1
            && repeated.canonical.events.size() == 1
            && repeated.canonical.people.front().track_id != first_sequence_id
            && repeated.canonical.events.front().status == second.canonical.events.front().status,
        "Reset did not clear analytics state while preserving unique track IDs");
    auto unsupported = syntheticPpeFrame(3, 300);
    unsupported.contract_version = "2.0.0";
    requireThrows([&] { pipeline.process(unsupported); },
        "Unsupported contract version was accepted");

    AnalyticsPipeline accepted_source(config);
    auto accepted = syntheticPpeFrame(1, 100);
    accepted.source_id = "ZONE/A 01";
    accepted_source.process(accepted);
    for (const std::string source : {
             "prefix-rtsp://camera", "prefix-rtsps://camera", "camera-password", "name@host"}) {
        AnalyticsPipeline rejected_source(config);
        auto forbidden = syntheticPpeFrame(1, 100);
        forbidden.source_id = source;
        requireThrows([&] { rejected_source.process(forbidden); },
            "Secret-like source ID was accepted: " + source);
    }
}

}  // namespace

int main() {
    const std::vector<std::pair<std::string, std::function<void()>>> tests{
        {"engine metadata prefix", testEngineRawAndMetadataPrefix},
        {"letterbox mapping", testLetterboxMappingAndPacking},
        {"detect decode and NMS", testDetectSchemaDecodeAndNms},
        {"pose decode", testPoseSchemaAndDecode},
        {"CLI URLs and invariants", testCliUrlsAndInvariantDefense},
        {"compute selection and probe contract", testComputeSelectionAndProbeContract},
        {"runtime execution planning", testRuntimeExecutionPlanning},
        {"ByteTrack lifecycle", testByteTrackLifecycleAndLowConfidenceAssociation},
        {"ByteTrack bounds and identity", testByteTrackCapacityResetEmptyFramesAndTies},
        {"PPE analytics", testPpeAssociationVotingAndCooldown},
        {"fall analytics", testFallConfirmationRecoveryAndCooldown},
        {"pose filtering tracker compatibility", testPoseFilteringPreservesTrackerIdentity},
        {"canonical deterministic pipeline", testCanonicalContractsAndDeterministicPipeline},
    };
    int failures = 0;
    for (const auto& [name, test] : tests) {
        try {
            test();
            std::cout << "PASS: " << name << '\n';
        } catch (const std::exception& error) {
            ++failures;
            std::cerr << "FAIL: " << name << ": " << error.what() << '\n';
        }
    }
    std::cout << (tests.size() - static_cast<std::size_t>(failures)) << "/"
              << tests.size() << " CPU tests passed\n";
    return failures == 0 ? 0 : 1;
}
