// SPDX-License-Identifier: AGPL-3.0-only

#include "cuajone/evidence.hpp"
#include "cuajone/ppe_analytics.hpp"

#include <opencv2/core.hpp>

#include <chrono>
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace cuajone;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
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

std::string pathUtf8(const std::filesystem::path& path) {
    const auto value = path.u8string();
    return {value.begin(), value.end()};
}

std::string csvEscape(const std::string& value) {
    if (value.find_first_of(",\"\r\n") == std::string::npos) return value;
    std::string result{"\""};
    for (const char character : value) {
        if (character == '"') result += "\"\"";
        else result += character;
    }
    return result + '"';
}

std::string readBytes(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

std::size_t countOccurrences(const std::string& value, const std::string& needle) {
    std::size_t count{};
    std::size_t position{};
    while ((position = value.find(needle, position)) != std::string::npos) {
        ++count;
        position += needle.size();
    }
    return count;
}

std::size_t regularFileCount(const std::filesystem::path& path) {
    std::size_t count{};
    if (!std::filesystem::exists(path)) return count;
    for (const auto& entry : std::filesystem::directory_iterator(path)) {
        if (entry.is_regular_file()) ++count;
    }
    return count;
}

class TemporaryDirectory {
public:
    explicit TemporaryDirectory(std::string name) {
        const auto token = std::chrono::steady_clock::now().time_since_epoch().count();
        path_ = std::filesystem::temp_directory_path()
            / ("cuajone,evidence-contract-" + std::move(name) + "-" + std::to_string(token));
        std::filesystem::create_directories(path_);
    }
    ~TemporaryDirectory() {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }
    const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
};

CanonicalEvent event(
    std::string id,
    std::string type,
    std::string time,
    std::string status,
    int track_id,
    float confidence,
    std::vector<PpeItem> missing = {},
    bool evaluated = true) {
    PpeEvaluation ppe;
    ppe.evaluated = evaluated;
    ppe.compliant = evaluated && missing.empty();
    ppe.samples = evaluated ? 12 : 1;
    for (const PpeItem item : requiredPpeItems()) {
        const bool present = evaluated
            && std::find(missing.begin(), missing.end(), item) == missing.end();
        ppe.items.push_back({item, true, present, present ? 0.8F : 0.1F,
            present ? 0.9F : 0.0F, std::nullopt});
    }
    return {
        std::move(id), "urn:cuajone:camera:CAM_01", std::move(type), std::move(time),
        "track/" + std::to_string(track_id), 1, 100, track_id, std::move(status), confidence, ppe,
    };
}

void testExactContractAndAppendBehavior() {
    TemporaryDirectory temporary("append");
    const cv::Mat frame(24, 32, CV_8UC3, cv::Scalar(10, 20, 30));
    const std::string camera = "CAM/01, \"Norte\"";
    const auto first_event = event(
        "evt-CAM-12345678", "com.cuajone.safety.ppe.violation.v2",
        "2026-01-02T03:04:05.678901Z", "Falta: Vest", 42, 0.875F, {PpeItem::Vest});

    EvidenceWriter writer(temporary.path());
    const EvidenceRecord first = writer.append(frame, camera, first_event);
    const auto csv = temporary.path() / "Reporte_Eventos_Seguridad_v2.csv";
    const auto images = temporary.path() / "Evidencias";
    require(std::filesystem::is_regular_file(first.image_path),
        "Evidence image was not created");
    require(first.image_path.filename()
            == "CAM_01___Norte__INCUMPLIMIENTO_EPP_20260102_030405_678_12345678.jpg",
        "Evidence filename does not follow the operator semantic pattern");

    const std::string first_content = readBytes(csv);
    require(first_content.starts_with(std::string(kOperatorCsvHeader) + "\n")
            && first_content.find("2.0.0,evt-CAM-12345678") != std::string::npos
            && first_content.find("INCUMPLIMIENTO_EPP,SI,SI,NO,SI,SI,SI,SI,Vest,Falta: Vest") != std::string::npos,
        "Initial v2 operator CSV does not expose all seven PPE states");

    const EvidenceRecord second = writer.append(frame, "CAM_01", event(
        "evt-CAM-ABCDEFGH", "com.cuajone.safety.fall.possible.v2",
        "2026-01-02T03:04:06Z", "Evaluando EPP", 7, 0.9F, {}, false));
    require(std::all_of(second.ppe_states.begin(), second.ppe_states.end(),
                [](const std::string& value) { return value == "N/D"; })
            && second.ppe_status == "En evaluaci\xC3\xB3n",
        "Unknown PPE state did not preserve Python SI/NO/N/D semantics");
    require(second.image_path.filename()
            == "CAM_01_POSIBLE_CAIDA_20260102_030406_000_ABCDEFGH.jpg",
        "Fall evidence filename changed");

    EvidenceWriter reopened(temporary.path());
    const EvidenceRecord third = reopened.append(frame, "CAM_01", event(
        "evt-CAM-IJKLMNOP", "com.cuajone.safety.ppe.violation.v2",
        "2026-01-02T03:04:07.1Z", "Falta: Hard_hat", 8, 1.0F, {PpeItem::HardHat}));
    require(third.ppe_states[static_cast<std::size_t>(PpeItem::HardHat)] == "NO"
            && third.ppe_states[static_cast<std::size_t>(PpeItem::Vest)] == "SI",
        "PPE v2 item mapping drifted");

    const std::string content = readBytes(csv);
    require(countOccurrences(content, std::string(kOperatorCsvHeader)) == 1,
        "CSV header was duplicated after append/reopen");
    require(countOccurrences(content, "PENDIENTE,,,") == 3,
        "CSV append did not preserve all operator rows");
    require(content.find(
                "POSIBLE_CAIDA,N/D,N/D,N/D,N/D,N/D,N/D,N/D,,En evaluaci\xC3\xB3n")
            != std::string::npos,
        "Fall row fields or UTF-8 encoding changed");
    require(regularFileCount(images) == 3, "Append did not retain one image per row");
}

void testMalformedTimestampDoesNotWrite() {
    TemporaryDirectory temporary("timestamp");
    EvidenceWriter writer(temporary.path());
    const cv::Mat frame(8, 8, CV_8UC3, cv::Scalar(1, 2, 3));
    requireThrows([&] {
        writer.append(frame, "CAM_01", event(
            "evt-invalid-time", "com.cuajone.safety.ppe.violation.v2",
            "2026-02-30T25:61:00Z", "Falta Casco", 1, 0.5F));
    }, "Malformed canonical timestamp was accepted");
    require(!std::filesystem::exists(
            temporary.path() / "Reporte_Eventos_Seguridad_v2.csv"),
        "Malformed timestamp created a CSV");
    require(regularFileCount(temporary.path() / "Evidencias") == 0,
        "Malformed timestamp created evidence");
}

void testImageFailureDoesNotCreatePartialCsvRow() {
    TemporaryDirectory temporary("image-failure");
    EvidenceWriter writer(temporary.path());
    requireThrows([&] {
        writer.append(cv::Mat{}, "CAM_01", event(
            "evt-empty-frame", "com.cuajone.safety.fall.possible.v2",
            "2026-01-02T03:04:05.000Z", "En evaluaci\xC3\xB3n", 2, 0.4F));
    }, "Empty evidence frame was accepted");
    require(!std::filesystem::exists(
            temporary.path() / "Reporte_Eventos_Seguridad_v2.csv"),
        "Image failure created a partial CSV row");
    require(regularFileCount(temporary.path() / "Evidencias") == 0,
        "Image failure retained a temporary image");
}

void testCsvWriteFailureIsReported() {
    TemporaryDirectory temporary("csv-failure");
    EvidenceWriter writer(temporary.path());
    std::filesystem::create_directory(
        temporary.path() / "Reporte_Eventos_Seguridad_v2.csv");
    const cv::Mat frame(8, 8, CV_8UC3, cv::Scalar(1, 2, 3));
    requireThrows([&] {
        writer.append(frame, "CAM_01", event(
            "evt-csv-failure", "com.cuajone.safety.ppe.violation.v2",
            "2026-01-02T03:04:05.000Z", "Sin Casco y Chaleco", 3, 0.7F));
    }, "CSV path failure was not reported");
    require(!std::filesystem::is_regular_file(
            temporary.path() / "Reporte_Eventos_Seguridad_v2.csv"),
        "CSV failure produced an unexpected regular file");
}

void testWriterQueueFifoBlockingAndCloneLifetime() {
    TemporaryDirectory temporary("queue-fifo");
    std::mutex mutex;
    std::vector<std::string> ids;
    std::vector<const unsigned char*> frame_data;
    std::promise<void> first_started;
    std::promise<void> release_first;
    std::shared_future<void> release = release_first.get_future().share();
    std::atomic_bool first{true};
    EvidenceWriterQueue queue(temporary.path(), 1,
        [&](const cv::Mat& frame, const std::string&, const CanonicalEvent& value, std::string& stage) {
            stage = "v2";
            if (first.exchange(false)) {
                first_started.set_value();
                release.wait();
            }
            std::scoped_lock lock(mutex);
            ids.push_back(value.id);
            frame_data.push_back(frame.data);
            require(frame.at<cv::Vec3b>(0, 0) == cv::Vec3b(10, 20, 30),
                "Queued evidence did not retain its deep-cloned annotated pixels");
        });
    cv::Mat frame(8, 8, CV_8UC3, cv::Scalar(10, 20, 30));
    const auto clone = EvidenceWriterQueue::cloneAnnotatedFrame(frame);
    require(queue.enqueue(clone, "CAM_01", event(
        "evt-fifo-1", "com.cuajone.safety.fall.possible.v2", "2026-01-02T03:04:05Z", "fall", 1, 0.8F)),
        "First queued event was rejected");
    first_started.get_future().wait();
    require(queue.enqueue(clone, "CAM_01", event(
        "evt-fifo-2", "com.cuajone.safety.fall.possible.v2", "2026-01-02T03:04:06Z", "fall", 2, 0.8F)),
        "Second queued event was rejected");
    frame.setTo(cv::Scalar(99, 99, 99));
    std::promise<void> enqueue_entered;
    std::thread blocked([&] {
        enqueue_entered.set_value();
        const bool accepted = queue.enqueue(clone, "CAM_01", event(
            "evt-fifo-3", "com.cuajone.safety.fall.possible.v2", "2026-01-02T03:04:07Z", "fall", 3, 0.8F));
        require(accepted, "Full queue dropped the newest event");
    });
    enqueue_entered.get_future().wait();
    while (queue.stats().blocked_enqueue_count == 0) std::this_thread::yield();
    release_first.set_value();
    blocked.join();
    queue.drainAndStop();
    const auto stats = queue.stats();
    require(ids == std::vector<std::string>{"evt-fifo-1", "evt-fifo-2", "evt-fifo-3"}
            && frame_data.size() == 3 && frame_data[0] == frame_data[1] && frame_data[1] == frame_data[2],
        "Queue did not preserve FIFO order or share one immutable frame clone");
    require(stats.accepted == 3 && stats.written == 3 && stats.failed == 0
            && stats.high_water_depth == 1 && stats.blocked_enqueue_count == 1
            && stats.current_depth == 0,
        "Queue FIFO/block/drain accounting changed");
}

void testWriterQueueTerminalFailuresAreAccounted() {
    for (const std::string stage : {"v2", "v3"}) {
        TemporaryDirectory temporary("queue-" + stage);
        std::promise<void> first_started;
        std::promise<void> release_first;
        std::shared_future<void> release = release_first.get_future().share();
        EvidenceWriterQueue queue(temporary.path(), 2,
            [&](const cv::Mat&, const std::string&, const CanonicalEvent&, std::string& current_stage) {
                current_stage = stage;
                first_started.set_value();
                release.wait();
                throw std::runtime_error("synthetic " + stage + " failure");
            });
        const auto clone = EvidenceWriterQueue::cloneAnnotatedFrame(
            cv::Mat(4, 4, CV_8UC3, cv::Scalar(1, 2, 3)));
        require(queue.enqueue(clone, "CAM_01", event(
            "evt-" + stage + "-1", "com.cuajone.safety.ppe.violation.v2", "2026-01-02T03:04:05Z", "missing", 1, 0.8F)),
            "First failing job was not accepted");
        first_started.get_future().wait();
        require(queue.enqueue(clone, "CAM_01", event(
            "evt-" + stage + "-2", "com.cuajone.safety.ppe.violation.v2", "2026-01-02T03:04:06Z", "missing", 2, 0.8F)),
            "Accepted job before terminal failure was rejected");
        release_first.set_value();
        queue.drainAndStop();
        const auto stats = queue.stats();
        require(stats.accepted == 2 && stats.written == 0 && stats.failed == 2 && stats.terminal_failure,
            "Terminal writer failure did not account for every accepted job");
        require(!queue.enqueue(clone, "CAM_01", event(
                    "evt-" + stage + "-3", "com.cuajone.safety.ppe.violation.v2", "2026-01-02T03:04:07Z", "missing", 3, 0.8F)),
            "Terminal queue accepted a later job");
        const std::string ledger = readBytes(temporary.path() / "evidence_writer_failures.jsonl");
        require(ledger.find("\"stage\":\"" + stage + "\"") != std::string::npos
                && ledger.find("\"stage\":\"terminal_rejected\"") != std::string::npos,
            "Failure ledger omitted writer stage or accepted terminal rejection");
    }
}

void testWriterQueueRedactsImageFailureDetails() {
    TemporaryDirectory temporary("queue-redaction");
    const std::string secret{"rtsp-user:password-123@example.invalid"};
    const std::string event_id{"evt-redaction-12345678"};
    EvidenceWriterQueue queue(temporary.path(), 1,
        [&](const cv::Mat&, const std::string& source_label, const CanonicalEvent& value,
            std::string& stage) {
            stage = "v2";
            EvidenceWriter writer(temporary.path());
            writer.append(cv::Mat{}, source_label, value);
        });
    const auto clone = EvidenceWriterQueue::cloneAnnotatedFrame(
        cv::Mat(4, 4, CV_8UC3, cv::Scalar(1, 2, 3)));
    require(queue.enqueue(clone, secret, event(
                event_id, "com.cuajone.safety.ppe.violation.v2", "2026-01-02T03:04:05Z",
                "missing", 1, 0.8F)),
        "Failing image-write job was not accepted");
    queue.drainAndStop();

    const auto stats = queue.stats();
    const std::string ledger = readBytes(temporary.path() / "evidence_writer_failures.jsonl");
    const std::string console_safe = queue.failureMessage();
    require(stats.accepted == 1 && stats.written == 0 && stats.failed == 1 && stats.terminal_failure,
        "Image write failure changed queue accounting");
    require(ledger.find("\"event_id\":\"" + event_id + "\"") != std::string::npos
            && ledger.find("\"stage\":\"v2\"") != std::string::npos
            && ledger.find("\"error\":\"evidence_write_failed\"") != std::string::npos,
        "Failure ledger omitted safe event, stage, or reason");
    require(console_safe.find(event_id) != std::string::npos
            && console_safe.find("v2") != std::string::npos
            && console_safe.find("evidence_write_failed") != std::string::npos,
        "Console-safe failure representation omitted event, stage, or reason");
    require(ledger.find(secret) == std::string::npos && console_safe.find(secret) == std::string::npos,
        "Failure reporting leaked the credential-like source label");
}

}  // namespace

int main() {
    const std::vector<std::pair<std::string, std::function<void()>>> tests{
        {"exact contract and append", testExactContractAndAppendBehavior},
        {"malformed timestamp", testMalformedTimestampDoesNotWrite},
        {"image failure", testImageFailureDoesNotCreatePartialCsvRow},
        {"CSV write failure", testCsvWriteFailureIsReported},
        {"writer queue FIFO/block/clone", testWriterQueueFifoBlockingAndCloneLifetime},
        {"writer queue terminal failures", testWriterQueueTerminalFailuresAreAccounted},
        {"writer queue redacts image failures", testWriterQueueRedactsImageFailureDetails},
    };
    int failures{};
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
              << tests.size() << " evidence tests passed\n";
    return failures == 0 ? 0 : 1;
}
