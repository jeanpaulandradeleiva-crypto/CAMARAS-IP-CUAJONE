// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>

namespace cuajone {

enum class PerformanceStage {
    FrameAge,
    PipelineTotal,
    PpePreprocess,
    PpeInference,
    PpeDecode,
    PosePreprocess,
    PoseInference,
    PoseDecode,
    Analytics,
    Render,
};

struct BenchmarkMetadata {
    std::size_t warmup_iterations{};
    std::size_t measured_iterations{};
    int image_width{};
    int image_height{};
};

struct BenchmarkProgress {
    bool warmup_complete{};
    std::size_t completed_measured_frames{};
    std::size_t measured_iterations{};
};

struct EvidenceQueueTelemetry {
    bool enabled{};
    std::size_t capacity{};
    std::uint64_t accepted{};
    std::uint64_t written{};
    std::uint64_t failed{};
    std::size_t current_depth{};
    std::size_t high_water_depth{};
    std::uint64_t blocked_enqueue_count{};
    std::chrono::steady_clock::duration blocked_enqueue_duration{};
    std::chrono::steady_clock::duration drain_duration{};
    bool terminal_failure{};
};

class PerformanceTelemetry {
public:
    static constexpr std::size_t kSampleCapacity = 256;

    explicit PerformanceTelemetry(std::string_view source_mode);
    ~PerformanceTelemetry();

    PerformanceTelemetry(const PerformanceTelemetry&) = delete;
    PerformanceTelemetry& operator=(const PerformanceTelemetry&) = delete;

    void addSample(PerformanceStage stage, std::chrono::steady_clock::duration duration);
    void capturedFrame();
    void processedFrame();
    void recordLatestSlotSequence(std::uint64_t previous_sequence, std::uint64_t latest_sequence);
    void skippedForTargetFps();
    void evidenceAppendAttempted();
    void evidenceAppendWritten();
    void evidenceAppendFailed();
    void setEvidenceQueueTelemetry(EvidenceQueueTelemetry telemetry);
    void reset();
    void setBenchmarkMetadata(BenchmarkMetadata metadata);
    [[nodiscard]] std::string jsonReport() const;

private:
    struct Impl;
    Impl* impl_;
};

template <typename Process>
void runBenchmarkIterations(
    std::size_t warmup_iterations,
    std::size_t measured_iterations,
    PerformanceTelemetry& telemetry,
    Process&& process) {
    for (std::size_t index = 0; index < warmup_iterations; ++index) process(index);
    telemetry.reset();
    for (std::size_t index = 0; index < measured_iterations; ++index) {
        process(warmup_iterations + index);
    }
}

template <typename Process, typename Progress>
void runBenchmarkIterations(
    std::size_t warmup_iterations,
    std::size_t measured_iterations,
    PerformanceTelemetry& telemetry,
    Process&& process,
    Progress&& progress) {
    for (std::size_t index = 0; index < warmup_iterations; ++index) process(index);
    telemetry.reset();
    progress(BenchmarkProgress{true, 0, measured_iterations});
    for (std::size_t index = 0; index < measured_iterations; ++index) {
        process(warmup_iterations + index);
        const std::size_t completed = index + 1;
        if (completed % 10 == 0) {
            progress(BenchmarkProgress{false, completed, measured_iterations});
        }
    }
}

std::string performanceSourceMode(std::string_view source);

}  // namespace cuajone
