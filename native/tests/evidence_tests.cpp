// SPDX-License-Identifier: AGPL-3.0-only

#include "cuajone/evidence.hpp"

#include <opencv2/core.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
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
    float confidence) {
    return {
        std::move(id), "urn:cuajone:camera:CAM_01", std::move(type), std::move(time),
        "track/" + std::to_string(track_id), 1, 100, track_id, std::move(status), confidence,
    };
}

void testExactContractAndAppendBehavior() {
    TemporaryDirectory temporary("append");
    const cv::Mat frame(24, 32, CV_8UC3, cv::Scalar(10, 20, 30));
    const std::string camera = "CAM/01, \"Norte\"";
    const auto first_event = event(
        "evt-CAM-12345678", "com.cuajone.safety.ppe.violation.v1",
        "2026-01-02T03:04:05.678901Z", "Falta Chaleco", 42, 0.875F);

    EvidenceWriter writer(temporary.path());
    const EvidenceRecord first = writer.append(frame, camera, first_event);
    const auto csv = temporary.path() / "Reporte_Eventos_Seguridad.csv";
    const auto images = temporary.path() / "Evidencias";
    require(std::filesystem::is_regular_file(first.image_path),
        "Evidence image was not created");
    require(first.image_path.filename()
            == "CAM_01___Norte__INCUMPLIMIENTO_EPP_20260102_030405_678_12345678.jpg",
        "Evidence filename does not follow the operator semantic pattern");

    const std::string first_row =
        "evt-CAM-12345678,\"CAM/01, \"\"Norte\"\"\",2026-01-02,03:04:05.678,"
        "INCUMPLIMIENTO_EPP,SI,NO,Falta Chaleco,0.875,42,PENDIENTE,,,"
        + csvEscape(pathUtf8(first.image_path)) + "\n";
    require(readBytes(csv) == std::string(kOperatorCsvHeader) + "\n" + first_row,
        "Initial operator CSV bytes differ from the exact contract");

    const EvidenceRecord second = writer.append(frame, "CAM_01", event(
        "evt-CAM-ABCDEFGH", "com.cuajone.safety.fall.possible.v1",
        "2026-01-02T03:04:06Z", "Evaluating PPE", 7, 0.9F));
    require(second.helmet == "N/D" && second.vest == "N/D"
            && second.ppe_status == "En evaluaci\xC3\xB3n",
        "Unknown PPE state did not preserve Python SI/NO/N/D semantics");
    require(second.image_path.filename()
            == "CAM_01_POSIBLE_CAIDA_20260102_030406_000_ABCDEFGH.jpg",
        "Fall evidence filename changed");

    EvidenceWriter reopened(temporary.path());
    const EvidenceRecord third = reopened.append(frame, "CAM_01", event(
        "evt-CAM-IJKLMNOP", "com.cuajone.safety.ppe.violation.v1",
        "2026-01-02T03:04:07.1Z", "Falta Casco", 8, 1.0F));
    require(third.helmet == "NO" && third.vest == "SI", "PPE status mapping drifted");

    const std::string content = readBytes(csv);
    require(countOccurrences(content, std::string(kOperatorCsvHeader)) == 1,
        "CSV header was duplicated after append/reopen");
    require(countOccurrences(content, "PENDIENTE,,,") == 3,
        "CSV append did not preserve all operator rows");
    require(content.find(
                "POSIBLE_CAIDA,N/D,N/D,En evaluaci\xC3\xB3n,0.9,7,PENDIENTE,,,")
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
            "evt-invalid-time", "com.cuajone.safety.ppe.violation.v1",
            "2026-02-30T25:61:00Z", "Falta Casco", 1, 0.5F));
    }, "Malformed canonical timestamp was accepted");
    require(!std::filesystem::exists(
            temporary.path() / "Reporte_Eventos_Seguridad.csv"),
        "Malformed timestamp created a CSV");
    require(regularFileCount(temporary.path() / "Evidencias") == 0,
        "Malformed timestamp created evidence");
}

void testImageFailureDoesNotCreatePartialCsvRow() {
    TemporaryDirectory temporary("image-failure");
    EvidenceWriter writer(temporary.path());
    requireThrows([&] {
        writer.append(cv::Mat{}, "CAM_01", event(
            "evt-empty-frame", "com.cuajone.safety.fall.possible.v1",
            "2026-01-02T03:04:05.000Z", "En evaluaci\xC3\xB3n", 2, 0.4F));
    }, "Empty evidence frame was accepted");
    require(!std::filesystem::exists(
            temporary.path() / "Reporte_Eventos_Seguridad.csv"),
        "Image failure created a partial CSV row");
    require(regularFileCount(temporary.path() / "Evidencias") == 0,
        "Image failure retained a temporary image");
}

void testCsvWriteFailureIsReported() {
    TemporaryDirectory temporary("csv-failure");
    EvidenceWriter writer(temporary.path());
    std::filesystem::create_directory(
        temporary.path() / "Reporte_Eventos_Seguridad.csv");
    const cv::Mat frame(8, 8, CV_8UC3, cv::Scalar(1, 2, 3));
    requireThrows([&] {
        writer.append(frame, "CAM_01", event(
            "evt-csv-failure", "com.cuajone.safety.ppe.violation.v1",
            "2026-01-02T03:04:05.000Z", "Sin Casco y Chaleco", 3, 0.7F));
    }, "CSV path failure was not reported");
    require(!std::filesystem::is_regular_file(
            temporary.path() / "Reporte_Eventos_Seguridad.csv"),
        "CSV failure produced an unexpected regular file");
}

}  // namespace

int main() {
    const std::vector<std::pair<std::string, std::function<void()>>> tests{
        {"exact contract and append", testExactContractAndAppendBehavior},
        {"malformed timestamp", testMalformedTimestampDoesNotWrite},
        {"image failure", testImageFailureDoesNotCreatePartialCsvRow},
        {"CSV write failure", testCsvWriteFailureIsReported},
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
