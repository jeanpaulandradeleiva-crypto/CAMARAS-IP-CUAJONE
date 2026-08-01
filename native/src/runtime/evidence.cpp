// SPDX-License-Identifier: AGPL-3.0-only

#include "cuajone/evidence.hpp"

#include <opencv2/imgcodecs.hpp>

#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace cuajone {
namespace {

std::string timestampNow(bool filename_safe) {
    const auto now = std::chrono::system_clock::now();
    const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    const std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::tm local{};
    localtime_s(&local, &time);
    std::ostringstream output;
    output << std::put_time(&local, filename_safe ? "%Y%m%d_%H%M%S" : "%Y-%m-%dT%H:%M:%S")
           << (filename_safe ? "_" : ".") << std::setfill('0') << std::setw(3) << milliseconds.count();
    return output.str();
}

std::string csvEscape(const std::string& value) {
    if (value.find_first_of(",\"\r\n") == std::string::npos) return value;
    std::string escaped{"\""};
    for (const char character : value) {
        if (character == '"') escaped += "\"\"";
        else escaped += character;
    }
    return escaped + "\"";
}

std::string safeFilePart(std::string value) {
    for (char& character : value) {
        const bool valid = (character >= 'a' && character <= 'z')
            || (character >= 'A' && character <= 'Z')
            || (character >= '0' && character <= '9') || character == '-' || character == '_';
        if (!valid) character = '_';
    }
    return value;
}

}  // namespace

void validateWritableOutput(const std::filesystem::path& output) {
    std::filesystem::create_directories(output);
    const auto probe = output / ".cuajone-write-probe.tmp";
    {
        std::ofstream stream(probe, std::ios::binary | std::ios::trunc);
        if (!stream) throw std::runtime_error("Output directory is not writable: " + output.string());
        stream << "probe";
        if (!stream) throw std::runtime_error("Output write probe failed: " + output.string());
    }
    std::error_code error;
    std::filesystem::remove(probe, error);
    if (error) throw std::runtime_error("Output write probe could not be removed: " + error.message());
}

EvidenceWriter::EvidenceWriter(std::filesystem::path output)
    : output_(std::move(output)), images_(output_ / "evidence"), csv_(output_ / "native_events.csv") {
    validateWritableOutput(output_);
    std::filesystem::create_directories(images_);
}

EvidenceRecord EvidenceWriter::append(
    const cv::Mat& annotated_frame,
    const std::string& source_label,
    const EventCandidate& event) {
    const std::string file_timestamp = timestampNow(true);
    const std::string filename = safeFilePart(source_label) + "_" + safeFilePart(event.event_type)
        + "_T" + std::to_string(event.track_id) + "_" + file_timestamp + "_"
        + std::to_string(++sequence_) + ".jpg";
    const auto image_path = images_ / filename;
    if (!cv::imwrite(image_path.string(), annotated_frame)) {
        throw std::runtime_error("OpenCV could not write evidence image: " + image_path.string());
    }

    const bool needs_header = !std::filesystem::exists(csv_) || std::filesystem::file_size(csv_) == 0;
    std::ofstream csv(csv_, std::ios::app);
    if (!csv) throw std::runtime_error("Cannot append native event CSV: " + csv_.string());
    if (needs_header) csv << "timestamp,source_label,track_id,event_type,confidence,status,image_path\n";
    EvidenceRecord record{
        timestampNow(false), source_label, event.track_id, event.event_type,
        event.confidence, event.status, image_path,
    };
    csv << csvEscape(record.timestamp) << ',' << csvEscape(record.source_label) << ','
        << record.track_id << ',' << csvEscape(record.event_type) << ','
        << std::fixed << std::setprecision(3) << record.confidence << ','
        << csvEscape(record.status) << ',' << csvEscape(record.image_path.string()) << '\n';
    csv.flush();
    if (!csv) throw std::runtime_error("Native event CSV write failed: " + csv_.string());
    return record;
}

}  // namespace cuajone
