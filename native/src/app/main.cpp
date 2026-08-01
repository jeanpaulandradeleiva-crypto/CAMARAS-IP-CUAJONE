// SPDX-License-Identifier: AGPL-3.0-only

#include "cuajone/capture.hpp"
#include "cuajone/cli.hpp"
#include "cuajone/engine_pipeline.hpp"
#include "cuajone/evidence.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace cuajone;
using Clock = std::chrono::steady_clock;

std::atomic_bool stop_requested{};

void requestStop(int) {
    stop_requested.store(true, std::memory_order_relaxed);
}

void validateSourceWithoutOpening(const std::string& source) {
    const bool network = isRtspSource(source);
    if (network) {
        validateRtspSource(source);
        return;
    }
    if (!std::filesystem::is_regular_file(source)) {
        throw std::runtime_error("Offline source does not exist: " + source);
    }
}

EnginePipelineConfig enginePipelineConfig(const RuntimeConfig& config, ComputeBackend backend) {
    return {
        backend,
        config.ppe_engine,
        config.pose_engine,
        config.ppe_onnx,
        config.pose_onnx,
        config.ppe_labels,
        config.pose_class_count,
        config.pose_keypoint_shape,
        config.allow_nonperson_pose_class,
        config.device,
        config.ppe_confidence,
        config.pose_confidence,
        config.nms_iou,
        config.max_det,
        {
            config.analytics_mode,
            {config.tracker_iou, config.tracker_max_age, config.tracker_max_tracks},
            config.ppe,
            config.fall,
            config.pose_confidence,
            config.nms_iou,
        },
    };
}

std::unique_ptr<NativeEnginePipeline> runBasePreflight(
    const RuntimeConfig& config,
    ComputeBackend backend) {
    validateSourceWithoutOpening(config.source);
    validateWritableOutput(config.output);
    auto pipeline = std::make_unique<NativeEnginePipeline>(enginePipelineConfig(config, backend));
    const auto& summary = pipeline->summary();
    std::cout << "OpenCV: " << CV_VERSION << " | provider: " << summary.provider << '\n';
    if (backend == ComputeBackend::Cuda) {
        std::cout << "CUDA device " << summary.device_index << ": " << summary.device_name
                  << " | SM " << summary.compute_major << '.' << summary.compute_minor
                  << " | devices: " << summary.device_count << '\n';
        std::cout << "PPE engine: " << config.ppe_engine.string()
                  << " | metadata prefix: " << (summary.ppe_metadata_prefix ? "yes" : "no") << '\n';
        if (summary.pose_loaded) {
            std::cout << "Pose engine: " << config.pose_engine.string()
                      << " | metadata prefix: " << (summary.pose_metadata_prefix ? "yes" : "no") << '\n';
        }
    } else {
        std::cout << "PPE ONNX: " << config.ppe_onnx.string() << '\n';
        if (summary.pose_loaded) std::cout << "Pose ONNX: " << config.pose_onnx.string() << '\n';
    }
    std::cout << "Source: " << redactSource(config.source) << " | label: " << config.source_label << '\n';
    std::cout << "Output: " << config.output.string() << '\n';
    return pipeline;
}

bool modelSetAvailable(
    const std::filesystem::path& ppe,
    const std::filesystem::path& pose,
    AnalyticsMode mode) {
    return std::filesystem::is_regular_file(ppe)
        && (mode == AnalyticsMode::PpeOnly || std::filesystem::is_regular_file(pose));
}

void drawPose(cv::Mat& frame, std::span<const Keypoint> keypoints, float threshold) {
    static constexpr std::array<std::array<int, 2>, 16> skeleton{{
        {0, 1}, {0, 2}, {1, 3}, {2, 4}, {5, 6}, {5, 7}, {7, 9}, {6, 8},
        {8, 10}, {5, 11}, {6, 12}, {11, 12}, {11, 13}, {13, 15}, {12, 14}, {14, 16},
    }};
    for (const auto& edge : skeleton) {
        if (edge[0] >= static_cast<int>(keypoints.size()) || edge[1] >= static_cast<int>(keypoints.size())) continue;
        const auto& from = keypoints[edge[0]];
        const auto& to = keypoints[edge[1]];
        if (from.confidence < threshold || to.confidence < threshold) continue;
        cv::line(frame, cv::Point(static_cast<int>(from.x), static_cast<int>(from.y)),
                 cv::Point(static_cast<int>(to.x), static_cast<int>(to.y)),
                 cv::Scalar(255, 140, 0), 2, cv::LINE_AA);
    }
}

void drawPerson(cv::Mat& frame, const CanonicalPerson& person) {
    cv::rectangle(
        frame,
        cv::Point(static_cast<int>(person.box.x1), static_cast<int>(person.box.y1)),
        cv::Point(static_cast<int>(person.box.x2), static_cast<int>(person.box.y2)),
        cv::Scalar(255, 255, 255), 2);
    std::string text = "T" + std::to_string(person.track_id) + " | " + person.ppe_status;
    if (person.fall_active) text += " | POSSIBLE FALL";
    cv::putText(frame, text,
        cv::Point(static_cast<int>(person.box.x1), std::max(25, static_cast<int>(person.box.y1) - 10)),
        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
}

std::string observedAtUtc() {
    const auto now = std::chrono::system_clock::now();
    const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    const std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::tm utc{};
    gmtime_s(&utc, &time);
    std::ostringstream output;
    output << std::put_time(&utc, "%Y-%m-%dT%H:%M:%S") << '.'
           << std::setfill('0') << std::setw(3) << milliseconds.count() << 'Z';
    return output.str();
}

void drawAssociatedItem(cv::Mat& frame, const std::optional<Detection>& item, const std::string& label) {
    if (!item) return;
    cv::rectangle(frame,
        cv::Point(static_cast<int>(item->box.x1), static_cast<int>(item->box.y1)),
        cv::Point(static_cast<int>(item->box.x2), static_cast<int>(item->box.y2)),
        cv::Scalar(0, 220, 255), 2);
    cv::putText(frame, label, cv::Point(static_cast<int>(item->box.x1), std::max(20, static_cast<int>(item->box.y1) - 6)),
        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 220, 255), 2, cv::LINE_AA);
}

int monitor(
    const RuntimeConfig& config,
    NativeEnginePipeline& pipeline) {
    EvidenceWriter evidence(config.output);
    LatestFrameCapture capture(
        config.source,
        std::chrono::duration<double>(config.reconnect_delay_seconds),
        std::chrono::duration<double>(config.maximum_reconnect_delay_seconds),
        config.capture_open_timeout,
        config.capture_read_timeout,
        config.rtsp_transport);

    capture.start();
    std::uint64_t sequence{};
    auto next_inference = Clock::time_point::min();
    std::optional<std::string> last_capture_error;
    bool first_inference_logged{};
    if (config.show_window) cv::namedWindow("Cuajone native analytics", cv::WINDOW_NORMAL);

    while (!stop_requested.load(std::memory_order_relaxed)) {
        cv::Mat frame;
        std::uint64_t latest_sequence = sequence;
        if (!capture.waitForLatest(sequence, frame, latest_sequence, std::chrono::milliseconds(100))) {
            if (capture.ended()) break;
            const auto error = capture.lastError();
            if (error && error != last_capture_error) {
                std::cerr << "Capture: " << *error << '\n';
                last_capture_error = error;
            }
            continue;
        }
        sequence = latest_sequence;
        const auto now = Clock::now();
        if (config.target_fps > 0.0 && now < next_inference) continue;
        if (config.target_fps > 0.0) {
            next_inference = now + std::chrono::duration_cast<Clock::duration>(
                std::chrono::duration<double>(1.0 / config.target_fps));
        }

        const auto monotonic_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()).count();
        const ProcessedFrame processed = pipeline.processFrame(
            frame, config.source_label, sequence, monotonic_ms, observedAtUtc());
        if (!first_inference_logged) {
            std::cout << "Inference: first frame processed | people: "
                      << processed.canonical.people.size() << '\n';
            first_inference_logged = true;
        }
        const float keypoint_threshold = std::clamp(config.pose_confidence, 0.25F, 0.50F);
        for (const auto& person : processed.canonical.people) {
            if (config.analytics_mode == AnalyticsMode::PpeFall) {
                drawPose(frame, person.keypoints, keypoint_threshold);
            }
            drawPerson(frame, person);
            const auto& association = processed.associations.at(person.track_id);
            drawAssociatedItem(frame, association.helmet_detection, "helmet");
            drawAssociatedItem(frame, association.vest_detection, "vest");
        }

        for (const auto& event : processed.canonical.events) {
            try {
                const EventCandidate candidate{
                    event.track_id,
                    event.type == "com.cuajone.safety.ppe.violation.v1"
                        ? "INCUMPLIMIENTO_EPP" : "POSIBLE_CAIDA",
                    event.status,
                    event.confidence,
                };
                const auto record = evidence.append(frame, config.source_label, candidate);
                std::cout << "Event: " << record.event_type << " | track " << record.track_id
                          << " | " << record.timestamp << '\n';
            } catch (const std::exception& error) {
                std::cerr << "Evidence write failed: " << error.what() << '\n';
            }
        }

        if (config.show_window) {
            cv::imshow("Cuajone native analytics", frame);
            const int key = cv::waitKey(1) & 0xFF;
            if (key == 'q' || key == 27
                || cv::getWindowProperty("Cuajone native analytics", cv::WND_PROP_VISIBLE) < 1.0) {
                stop_requested.store(true, std::memory_order_relaxed);
            }
        }
    }
    capture.stop();
    cv::destroyAllWindows();
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::signal(SIGINT, requestStop);
    std::signal(SIGTERM, requestStop);
#ifdef _WIN32
    std::signal(SIGBREAK, requestStop);
#endif
    try {
        RuntimeConfig config = parseCommandLine(argc, argv);
        if (config.help) {
            printHelp(std::cout);
            return 0;
        }
        if (config.hardware_probe_json) {
            const HardwareProbeResult probe = probeHardware();
            std::cout << hardwareProbeJson(probe) << '\n';
            return hardwareProbeExitCode(probe.status);
        }
        if (!config.compute_explicit) {
            if (const auto installed = installedComputeBackend()) config.compute_backend = *installed;
        }
        HardwareProbeStatus hardware_status = HardwareProbeStatus::NoNvidiaAdapter;
        if (config.compute_backend != ComputeBackend::Cpu) {
            const HardwareProbeResult probe = probeHardware();
            hardware_status = probe.status;
            std::cout << "Hardware probe: " << hardwareProbeStatusName(probe.status)
                      << " | " << hardwareProbeSummary(probe)
                      << " | " << probe.detail << '\n';
        }
        const bool tensor_rt_models = modelSetAvailable(
            config.ppe_engine, config.pose_engine, config.analytics_mode);
        const bool onnx_models = modelSetAvailable(
            config.ppe_onnx, config.pose_onnx, config.analytics_mode);
        const ComputeSelection selection = selectComputeBackend(config.compute_backend, {
            hardware_status,
            tensorRtBackendCompiled(),
            onnxCudaExecutionProviderCompiled(),
            tensor_rt_models,
            onnx_models,
        });
        std::cout << "Compute: " << computeBackendName(selection.backend)
                  << " | " << selection.reason << '\n';
        std::unique_ptr<NativeEnginePipeline> pipeline;
        ComputeBackend effective_backend = selection.backend;
        try {
            pipeline = runBasePreflight(config, effective_backend);
        } catch (const std::exception& cuda_error) {
            if (config.compute_backend != ComputeBackend::Auto
                || effective_backend != ComputeBackend::Cuda
                || !onnx_models) {
                throw;
            }
            std::cerr << "Auto CUDA validation failed; selecting CPU: " << cuda_error.what() << '\n';
            effective_backend = ComputeBackend::Cpu;
            pipeline = runBasePreflight(config, effective_backend);
        }
        std::cout << "Preflight: OK\n";
        if (config.preflight) return 0;
        return monitor(config, *pipeline);
    } catch (const std::invalid_argument& error) {
        std::cerr << "Configuration error: " << error.what() << "\nUse --help for usage.\n";
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "Runtime error: " << error.what() << '\n';
        return 1;
    }
}
