// SPDX-License-Identifier: AGPL-3.0-only

#include "cuajone/capture.hpp"
#include "cuajone/cli.hpp"
#include "cuajone/engine_pipeline.hpp"
#include "cuajone/evidence.hpp"
#include "cuajone/performance_telemetry.hpp"
#include "cuajone/runtime_execution_plan.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <csignal>
#include <ctime>
#include <cwchar>
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
constexpr char kLiveAnalyticsWindowTitle[] = "NexoAI Vision - Live Analytics";

#ifdef _WIN32
constexpr wchar_t kCudaWarmupChildEnvironment[] = L"CUAJONE_INTERNAL_CUDA_WARMUP_CHILD";

class UniqueHandle {
public:
    explicit UniqueHandle(HANDLE value = nullptr) noexcept : value_(value) {}
    ~UniqueHandle() { if (value_ != nullptr) CloseHandle(value_); }
    UniqueHandle(const UniqueHandle&) = delete;
    UniqueHandle& operator=(const UniqueHandle&) = delete;
    [[nodiscard]] HANDLE get() const noexcept { return value_; }

private:
    HANDLE value_;
};

bool isCudaWarmupChild() {
    return GetEnvironmentVariableW(kCudaWarmupChildEnvironment, nullptr, 0) != 0;
}

void runIsolatedCudaWarmup() {
    std::vector<wchar_t> executable(32768);
    const DWORD executable_size = GetModuleFileNameW(
        nullptr, executable.data(), static_cast<DWORD>(executable.size()));
    if (executable_size == 0 || executable_size >= executable.size()) {
        throw std::runtime_error("Could not resolve the runtime executable for CUDA warmup");
    }
    executable.resize(executable_size + 1);
    std::vector<wchar_t> command_line(
        GetCommandLineW(), GetCommandLineW() + std::wcslen(GetCommandLineW()) + 1);

    if (!SetEnvironmentVariableW(kCudaWarmupChildEnvironment, L"1")) {
        throw std::runtime_error("Could not configure the isolated CUDA warmup process");
    }
    PROCESS_INFORMATION process{};
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    const BOOL created = CreateProcessW(
        executable.data(), command_line.data(), nullptr, nullptr, FALSE,
        CREATE_NO_WINDOW, nullptr, nullptr, &startup, &process);
    const DWORD create_error = created ? ERROR_SUCCESS : GetLastError();
    SetEnvironmentVariableW(kCudaWarmupChildEnvironment, nullptr);
    if (!created) {
        throw std::runtime_error(
            "Could not start the isolated CUDA warmup process; Windows error "
            + std::to_string(create_error));
    }
    const UniqueHandle process_handle(process.hProcess);
    const UniqueHandle thread_handle(process.hThread);
    if (WaitForSingleObject(process_handle.get(), INFINITE) != WAIT_OBJECT_0) {
        throw std::runtime_error("Could not wait for the isolated CUDA warmup process");
    }
    DWORD exit_code{};
    if (!GetExitCodeProcess(process_handle.get(), &exit_code)) {
        throw std::runtime_error("Could not read the isolated CUDA warmup exit code");
    }
    if (exit_code != 0) {
        std::ostringstream message;
        message << "ONNX CUDA warmup failed in an isolated process with exit code 0x"
                << std::hex << std::uppercase << exit_code;
        throw std::runtime_error(message.str());
    }
}
#endif

void requestStop(int) {
    stop_requested.store(true, std::memory_order_relaxed);
}

#ifdef _WIN32
void applyLiveAnalyticsWindowIcon() {
    const HICON icon = LoadIconW(GetModuleHandleW(nullptr), MAKEINTRESOURCEW(101));
    if (icon == nullptr) return;
    const int length = MultiByteToWideChar(
        CP_UTF8, MB_ERR_INVALID_CHARS, kLiveAnalyticsWindowTitle, -1, nullptr, 0);
    if (length <= 0) return;
    std::wstring title(static_cast<std::size_t>(length), L'\0');
    if (MultiByteToWideChar(
            CP_UTF8, MB_ERR_INVALID_CHARS, kLiveAnalyticsWindowTitle, -1,
            title.data(), length) <= 0) {
        return;
    }
    HWND window = nullptr;
    while ((window = FindWindowExW(nullptr, window, nullptr, title.c_str())) != nullptr) {
        DWORD process_id{};
        GetWindowThreadProcessId(window, &process_id);
        if (process_id != GetCurrentProcessId()) continue;
        SendMessageW(window, WM_SETICON, ICON_SMALL, reinterpret_cast<LPARAM>(icon));
        SendMessageW(window, WM_SETICON, ICON_BIG, reinterpret_cast<LPARAM>(icon));
        return;
    }
}
#endif

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

EnginePipelineConfig enginePipelineConfig(
    const RuntimeConfig& config,
    const ComputeSelection& selection,
    PerformanceTelemetry* telemetry) {
    EnginePipelineConfig pipeline_config{
        selection.backend,
        selection.provider,
        config.ppe_engine,
        config.pose_engine,
        config.ppe_onnx,
        config.pose_onnx,
        config.ppe_labels,
        config.pose_class_count,
        config.pose_keypoint_shape,
        config.allow_nonperson_pose_class,
        config.device,
        config.image_size,
        config.ppe_confidence,
        config.ppe_class_confidences,
        config.ppe_enabled,
        config.pose_confidence,
        config.nms_iou,
        config.max_det,
        {
            config.analytics_mode,
            {
                config.tracker_high_threshold,
                0.10F,
                config.tracker_match_threshold,
                config.tracker_max_age,
                config.tracker_max_tracks,
                config.tracker_frame_rate,
            },
            config.ppe,
            config.fall,
            config.pose_confidence,
            config.nms_iou,
        },
        telemetry,
    };
#ifdef CUAJONE_INTERNAL_DIAGNOSTICS
    // This is deliberately unavailable in production binaries and only applies to offline benchmarks.
    char* serial_hybrid_benchmark{};
    std::size_t serial_hybrid_benchmark_length{};
    if (_dupenv_s(
            &serial_hybrid_benchmark, &serial_hybrid_benchmark_length,
            "CUAJONE_INTERNAL_SERIAL_HYBRID_BENCHMARK") != 0) {
        throw std::runtime_error("Could not read the internal serial hybrid benchmark switch");
    }
    pipeline_config.force_serial_hybrid = !config.benchmark_image.empty()
        && serial_hybrid_benchmark != nullptr && std::string_view(serial_hybrid_benchmark) == "1";
    std::free(serial_hybrid_benchmark);
#endif
    return pipeline_config;
}

std::unique_ptr<NativeEnginePipeline> runBasePreflight(
    const RuntimeConfig& config,
    const ComputeSelection& selection,
    PerformanceTelemetry* telemetry) {
    if (config.benchmark_image.empty()) {
        validateSourceWithoutOpening(config.source);
        validateWritableOutput(config.output);
    } else if (!std::filesystem::is_regular_file(config.benchmark_image)) {
        throw std::runtime_error("Benchmark image does not exist: " + config.benchmark_image.string());
    }
#ifdef _WIN32
    if (selection.provider == InferenceProvider::OnnxRuntimeCuda && !isCudaWarmupChild()
        && config.benchmark_image.empty()) {
        runIsolatedCudaWarmup();
    }
#endif
    auto pipeline = std::make_unique<NativeEnginePipeline>(enginePipelineConfig(config, selection, telemetry));
#ifdef _WIN32
    if (selection.provider == InferenceProvider::OnnxRuntimeCuda && isCudaWarmupChild()) {
        cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(32, 64, 96));
        const ProcessedFrame warmup = pipeline->processFrame(
            frame, "cuda-preflight", 0, 0, "1970-01-01T00:00:00Z");
        validateCanonicalMetadata(warmup.canonical);
        if (warmup.canonical.frame_width != frame.cols
            || warmup.canonical.frame_height != frame.rows) {
            throw std::runtime_error("ONNX CUDA warmup returned an invalid canonical frame");
        }
    }
#endif
    const auto& summary = pipeline->summary();
    std::cout << "OpenCV: " << CV_VERSION << " | provider: " << summary.provider << '\n';
    std::cout << "Inference imgsz: " << summary.image_size << 'x' << summary.image_size << '\n';
    if (selection.backend == ComputeBackend::Cuda) {
        std::cout << "CUDA device " << summary.device_index << ": " << summary.device_name
                   << " | SM " << summary.compute_major << '.' << summary.compute_minor
                   << " | devices: " << summary.device_count << '\n';
    }
    if (selection.provider == InferenceProvider::TensorRt) {
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
    if (config.benchmark_image.empty()) {
        std::cout << "Source: " << redactSource(config.source) << " | label: " << config.source_label << '\n';
        std::cout << "Output: " << config.output.string() << '\n';
    } else {
        std::cout << "Source: benchmark-image\n";
    }
    return pipeline;
}

ModelArtifactAvailability modelArtifactAvailability(const RuntimeConfig& config) {
    return {
        std::filesystem::is_regular_file(config.ppe_engine),
        std::filesystem::is_regular_file(config.pose_engine),
        std::filesystem::is_regular_file(config.ppe_onnx),
        std::filesystem::is_regular_file(config.pose_onnx),
    };
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
    if (!person.ppe) return;
    int line_y = static_cast<int>(person.box.y1) + 18;
    for (const auto& item : person.ppe->items) {
        if (!item.enabled) continue;
        const std::string state = !person.ppe->evaluated ? "..." : std::string(ppeWearStateName(item.wear_state));
        const cv::Scalar color = !person.ppe->evaluated || item.wear_state == PpeWearState::NotVerifiable
            ? cv::Scalar(0, 220, 255) : item.wear_state == PpeWearState::PresentCorrectly
            ? cv::Scalar(0, 220, 0) : cv::Scalar(0, 0, 255);
        cv::putText(frame,
            std::string(ppeItemLabel(item.item)) + ": " + state,
            cv::Point(static_cast<int>(person.box.x1) + 4, line_y),
            cv::FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv::LINE_AA);
        line_y += 17;
    }
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
    NativeEnginePipeline& pipeline,
    PerformanceTelemetry* telemetry) {
    EvidenceWriter evidence(config.output);
    EvidenceWriterV3 evidence_v3(config.output);
    LatestFrameCapture capture(
        config.source,
        std::chrono::duration<double>(config.reconnect_delay_seconds),
        std::chrono::duration<double>(config.maximum_reconnect_delay_seconds),
        config.capture_open_timeout,
        config.capture_read_timeout,
        config.rtsp_transport, telemetry);

    capture.start();
    std::uint64_t sequence{};
    auto next_inference = Clock::time_point::min();
    std::optional<std::string> last_capture_error;
    bool first_inference_logged{};
    if (config.show_window) {
        cv::namedWindow(kLiveAnalyticsWindowTitle, cv::WINDOW_NORMAL);
#ifdef _WIN32
        applyLiveAnalyticsWindowIcon();
#endif
    }

    while (!stop_requested.load(std::memory_order_relaxed)) {
        cv::Mat frame;
        std::uint64_t latest_sequence = sequence;
        Clock::time_point published_at;
        if (!capture.waitForLatest(sequence, frame, latest_sequence, published_at, std::chrono::milliseconds(100))) {
            if (capture.ended()) break;
            const auto error = capture.lastError();
            if (error && error != last_capture_error) {
                std::cerr << "Capture: " << *error << '\n';
                last_capture_error = error;
            }
            continue;
        }
        if (telemetry != nullptr) telemetry->recordLatestSlotSequence(sequence, latest_sequence);
        sequence = latest_sequence;
        const auto now = Clock::now();
        if (telemetry != nullptr) telemetry->addSample(PerformanceStage::FrameAge, now - published_at);
        if (config.target_fps > 0.0 && now < next_inference) {
            if (telemetry != nullptr) telemetry->skippedForTargetFps();
            continue;
        }
        if (config.target_fps > 0.0) {
            next_inference = now + std::chrono::duration_cast<Clock::duration>(
                std::chrono::duration<double>(1.0 / config.target_fps));
        }

        const auto monotonic_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()).count();
        const ProcessedFrame processed = pipeline.processFrame(
            frame, config.source_label, sequence, monotonic_ms, observedAtUtc());
        if (telemetry != nullptr) telemetry->processedFrame();
        if (!first_inference_logged) {
            std::cout << "Inference: first frame processed | people: "
                      << processed.canonical.people.size() << '\n';
            first_inference_logged = true;
        }
        const auto render_started = telemetry == nullptr ? Clock::time_point{} : Clock::now();
        const float keypoint_threshold = std::clamp(config.pose_confidence, 0.25F, 0.50F);
        for (const auto& person : processed.canonical.people) {
            if (config.analytics_mode == AnalyticsMode::PpeFall) {
                drawPose(frame, person.keypoints, keypoint_threshold);
            }
            drawPerson(frame, person);
            const auto& association = processed.associations.at(person.track_id);
            for (const PpeItem item : requiredPpeItems()) {
                if (!config.ppe_enabled[static_cast<std::size_t>(item)]) continue;
                drawAssociatedItem(frame, association.detection(item), std::string(ppeItemLabel(item)));
            }
        }
        if (telemetry != nullptr) telemetry->addSample(PerformanceStage::Render, Clock::now() - render_started);

        for (const auto& event : processed.canonical.events) {
            try {
                if (telemetry != nullptr) telemetry->evidenceAppendAttempted();
                const auto record = evidence.append(frame, config.source_label, event);
                if (event.type == "com.cuajone.safety.ppe.violation.v2") evidence_v3.append(event);
                if (telemetry != nullptr) telemetry->evidenceAppendWritten();
                std::cout << "Event: " << record.event_type << " | track " << record.track_id
                          << " | " << record.date << 'T' << record.time << "Z\n";
            } catch (const std::exception& error) {
                if (telemetry != nullptr) telemetry->evidenceAppendFailed();
                std::cerr << "Evidence write failed: " << error.what() << '\n';
            }
        }

        if (config.show_window) {
            cv::imshow(kLiveAnalyticsWindowTitle, frame);
            const int key = cv::waitKey(1) & 0xFF;
            if (key == 'q' || key == 27
                || cv::getWindowProperty(kLiveAnalyticsWindowTitle, cv::WND_PROP_VISIBLE) < 1.0) {
                stop_requested.store(true, std::memory_order_relaxed);
            }
        }
    }
    capture.stop();
    cv::destroyAllWindows();
    return 0;
}

std::string benchmarkObservedAt(std::uint64_t frame_id) {
    const std::uint64_t seconds = frame_id / 1000;
    const std::uint64_t milliseconds = frame_id % 1000;
    std::ostringstream output;
    output << "1970-01-01T00:00:" << std::setfill('0') << std::setw(2) << seconds
           << '.' << std::setw(3) << milliseconds << 'Z';
    return output.str();
}

int benchmark(
    const RuntimeConfig& config,
    NativeEnginePipeline& pipeline,
    PerformanceTelemetry& telemetry) {
    const cv::Mat image = cv::imread(config.benchmark_image.string(), cv::IMREAD_COLOR);
    if (image.empty()) throw std::runtime_error("Could not decode benchmark image");
    telemetry.setBenchmarkMetadata({
        config.benchmark_warmup,
        config.benchmark_iterations,
        image.cols,
        image.rows,
    });
    runBenchmarkIterations(
        config.benchmark_warmup, config.benchmark_iterations, telemetry,
        [&](std::size_t iteration) {
            const std::uint64_t frame_id = static_cast<std::uint64_t>(iteration) + 1;
            const auto timestamp_ms = static_cast<std::int64_t>(frame_id);
            static_cast<void>(pipeline.processFrame(
                image, "benchmark-image", frame_id, timestamp_ms, benchmarkObservedAt(frame_id)));
            telemetry.processedFrame();
        },
        [](const BenchmarkProgress& progress) {
            if (progress.warmup_complete) {
                std::cout << "Benchmark progress: warmup complete; measured frames 0/"
                          << progress.measured_iterations << '\n';
                return;
            }
            std::cout << "Benchmark progress: measured frames "
                      << progress.completed_measured_frames << '/'
                      << progress.measured_iterations << '\n';
        });
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
        std::optional<ComputeBackend> installed_backend;
        if (!config.compute_explicit) installed_backend = installedComputeBackend();
        const ComputeBackend requested_backend = resolveRequestedComputeBackend(
            config.compute_backend, config.compute_explicit, installed_backend);
        HardwareProbeStatus hardware_status = HardwareProbeStatus::NoNvidiaAdapter;
        if (requested_backend != ComputeBackend::Cpu) {
            const HardwareProbeResult probe = probeHardware();
            hardware_status = probe.status;
            std::cout << "Hardware probe: " << hardwareProbeStatusName(probe.status)
                      << " | " << hardwareProbeSummary(probe)
                      << " | " << probe.detail << '\n';
        }
        const RuntimeExecutionPlan plan = planRuntimeExecution({
            config.compute_backend,
            config.compute_explicit,
            installed_backend,
            config.analytics_mode,
            {
                hardware_status,
                tensorRtBackendCompiled(),
                onnxCudaExecutionProviderCompiled(),
            },
            modelArtifactAvailability(config),
        });
        std::cout << "Compute: " << computeBackendName(plan.selection.backend)
                  << " | " << plan.selection.reason << '\n';
        std::unique_ptr<NativeEnginePipeline> pipeline;
        std::unique_ptr<PerformanceTelemetry> telemetry;
        if (config.performance_report) {
            telemetry = std::make_unique<PerformanceTelemetry>(
                config.benchmark_image.empty() ? performanceSourceMode(config.source) : "benchmark-image");
        }
        ComputeSelection effective_selection = plan.selection;
        try {
            pipeline = runBasePreflight(config, effective_selection, telemetry.get());
        } catch (const std::exception& cuda_error) {
            if (!plan.preflight_failure_fallback) {
                throw;
            }
            std::cerr << "Auto CUDA validation failed; selecting CPU: " << cuda_error.what() << '\n';
            effective_selection = *plan.preflight_failure_fallback;
            pipeline = runBasePreflight(config, effective_selection, telemetry.get());
        }
        std::cout << "Preflight: OK\n";
#ifdef _WIN32
        if (isCudaWarmupChild()) return 0;
#endif
        if (config.preflight) return 0;
        const int result = config.benchmark_image.empty()
            ? monitor(config, *pipeline, telemetry.get())
            : benchmark(config, *pipeline, *telemetry);
        if (telemetry) std::cout << telemetry->jsonReport() << '\n';
        return result;
    } catch (const std::invalid_argument& error) {
        std::cerr << "Configuration error: " << error.what() << "\nUse --help for usage.\n";
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "Runtime error: " << error.what() << '\n';
        return 1;
    }
}
