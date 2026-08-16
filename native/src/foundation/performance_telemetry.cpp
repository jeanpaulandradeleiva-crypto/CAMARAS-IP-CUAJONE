// SPDX-License-Identifier: AGPL-3.0-only

#include "cuajone/performance_telemetry.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <vector>

namespace cuajone {
namespace {

constexpr std::array<std::string_view, 10> kStageNames{
    "frame_age", "pipeline_total", "ppe_preprocess", "ppe_inference", "ppe_decode",
    "pose_preprocess", "pose_inference", "pose_decode", "analytics", "render",
};

constexpr std::size_t stageIndex(PerformanceStage stage) {
    return static_cast<std::size_t>(stage);
}

struct RollingSamples {
    std::array<double, PerformanceTelemetry::kSampleCapacity> values{};
    std::size_t size{};
    std::size_t next{};

    void add(double value) {
        values[next] = value;
        next = (next + 1) % values.size();
        size = std::min(size + 1, values.size());
    }

    void reset() noexcept { size = 0; next = 0; }
};

void appendStats(std::ostringstream& output, const RollingSamples& samples) {
    std::vector<double> sorted(samples.values.begin(), samples.values.begin() + samples.size);
    std::sort(sorted.begin(), sorted.end());
    const auto percentile = [&sorted](double quantile) {
        if (sorted.empty()) return 0.0;
        const std::size_t index = static_cast<std::size_t>(std::ceil(quantile * sorted.size())) - 1;
        return sorted[index];
    };
    output << "{\"samples\":" << samples.size
           << ",\"p50_ms\":" << percentile(0.50)
           << ",\"p95_ms\":" << percentile(0.95)
           << ",\"p99_ms\":" << percentile(0.99) << '}';
}

}  // namespace

struct PerformanceTelemetry::Impl {
    explicit Impl(std::string_view mode) : source_mode(mode) {}

    std::string source_mode;
    mutable std::mutex mutex;
    std::array<RollingSamples, kStageNames.size()> stages;
    std::uint64_t captured_frames{};
    std::uint64_t processed_frames{};
    std::uint64_t latest_slot_sequence_gap_drops{};
    std::uint64_t target_fps_skipped_frames{};
    std::uint64_t evidence_append_attempted{};
    std::uint64_t evidence_append_written{};
    std::uint64_t evidence_append_failed{};
    std::optional<BenchmarkMetadata> benchmark;
};

PerformanceTelemetry::PerformanceTelemetry(std::string_view source_mode)
    : impl_(new Impl(source_mode)) {}

PerformanceTelemetry::~PerformanceTelemetry() {
    delete impl_;
}

void PerformanceTelemetry::addSample(PerformanceStage stage, std::chrono::steady_clock::duration duration) {
    const double milliseconds = std::chrono::duration<double, std::milli>(duration).count();
    std::scoped_lock lock(impl_->mutex);
    impl_->stages[stageIndex(stage)].add(milliseconds);
}

void PerformanceTelemetry::capturedFrame() {
    std::scoped_lock lock(impl_->mutex);
    ++impl_->captured_frames;
}

void PerformanceTelemetry::processedFrame() {
    std::scoped_lock lock(impl_->mutex);
    ++impl_->processed_frames;
}

void PerformanceTelemetry::recordLatestSlotSequence(
    std::uint64_t previous_sequence,
    std::uint64_t latest_sequence) {
    if (latest_sequence <= previous_sequence + 1) return;
    std::scoped_lock lock(impl_->mutex);
    impl_->latest_slot_sequence_gap_drops += latest_sequence - previous_sequence - 1;
}

void PerformanceTelemetry::skippedForTargetFps() {
    std::scoped_lock lock(impl_->mutex);
    ++impl_->target_fps_skipped_frames;
}

void PerformanceTelemetry::evidenceAppendAttempted() {
    std::scoped_lock lock(impl_->mutex);
    ++impl_->evidence_append_attempted;
}

void PerformanceTelemetry::evidenceAppendWritten() {
    std::scoped_lock lock(impl_->mutex);
    ++impl_->evidence_append_written;
}

void PerformanceTelemetry::evidenceAppendFailed() {
    std::scoped_lock lock(impl_->mutex);
    ++impl_->evidence_append_failed;
}

void PerformanceTelemetry::reset() {
    std::scoped_lock lock(impl_->mutex);
    for (auto& stage : impl_->stages) stage.reset();
    impl_->captured_frames = 0;
    impl_->processed_frames = 0;
    impl_->latest_slot_sequence_gap_drops = 0;
    impl_->target_fps_skipped_frames = 0;
    impl_->evidence_append_attempted = 0;
    impl_->evidence_append_written = 0;
    impl_->evidence_append_failed = 0;
}

void PerformanceTelemetry::setBenchmarkMetadata(BenchmarkMetadata metadata) {
    std::scoped_lock lock(impl_->mutex);
    impl_->benchmark = metadata;
}

std::string PerformanceTelemetry::jsonReport() const {
    std::scoped_lock lock(impl_->mutex);
    std::ostringstream output;
    output << std::fixed << std::setprecision(3);
    output << "{\"schema_version\":1,\"source_mode\":\"" << impl_->source_mode
           << "\",\"counts\":{\"captured_frames\":" << impl_->captured_frames
           << ",\"processed_frames\":" << impl_->processed_frames
           << ",\"latest_slot_sequence_gap_drops\":" << impl_->latest_slot_sequence_gap_drops
           << ",\"target_fps_skipped_frames\":" << impl_->target_fps_skipped_frames
           << "},\"stages_ms\":{";
    for (std::size_t index = 0; index < kStageNames.size(); ++index) {
        if (index != 0) output << ',';
        output << '"' << kStageNames[index] << "\":";
        appendStats(output, impl_->stages[index]);
    }
    output << "},\"evidence\":{\"append_attempted\":" << impl_->evidence_append_attempted
            << ",\"append_written\":" << impl_->evidence_append_written
            << ",\"append_failed\":" << impl_->evidence_append_failed << '}';
    if (impl_->benchmark) {
        const auto& benchmark = *impl_->benchmark;
        output << ",\"benchmark\":{\"warmup_iterations\":" << benchmark.warmup_iterations
               << ",\"measured_iterations\":" << benchmark.measured_iterations
               << ",\"image_width\":" << benchmark.image_width
               << ",\"image_height\":" << benchmark.image_height
               << ",\"retained_sample_capacity\":" << kSampleCapacity
               << ",\"retained_sample_count\":"
               << std::min(benchmark.measured_iterations, kSampleCapacity) << '}';
    }
    output << '}';
    return output.str();
}

std::string performanceSourceMode(std::string_view source) {
    if (source.starts_with("rtsp://") || source.starts_with("rtsps://")) return "rtsp";
    const std::size_t dot = source.rfind('.');
    if (dot != std::string_view::npos) {
        std::string extension(source.substr(dot));
        std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char value) {
            return static_cast<char>(std::tolower(value));
        });
        if (extension == ".jpg" || extension == ".jpeg" || extension == ".png" || extension == ".bmp") {
            return "image";
        }
    }
    return "video";
}

}  // namespace cuajone
