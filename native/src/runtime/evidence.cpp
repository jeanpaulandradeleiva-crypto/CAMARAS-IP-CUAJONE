// SPDX-License-Identifier: AGPL-3.0-only

#include "cuajone/evidence.hpp"
#include "cuajone/ppe_analytics.hpp"

#include <opencv2/imgcodecs.hpp>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace cuajone {
namespace {

struct ParsedTimestamp {
    std::string date;
    std::string time;
    std::string filename;
};

bool allDigits(std::string_view value) {
    for (const char character : value) {
        if (character < '0' || character > '9') return false;
    }
    return !value.empty();
}

int decimal(std::string_view value) {
    int result{};
    for (const char character : value) result = result * 10 + (character - '0');
    return result;
}

ParsedTimestamp parseUtcTimestamp(const std::string& value) {
    const bool fixed_shape = value.size() >= 20
        && value[4] == '-' && value[7] == '-' && value[10] == 'T'
        && value[13] == ':' && value[16] == ':' && value.back() == 'Z'
        && allDigits(std::string_view(value).substr(0, 4))
        && allDigits(std::string_view(value).substr(5, 2))
        && allDigits(std::string_view(value).substr(8, 2))
        && allDigits(std::string_view(value).substr(11, 2))
        && allDigits(std::string_view(value).substr(14, 2))
        && allDigits(std::string_view(value).substr(17, 2));
    if (!fixed_shape) {
        throw std::invalid_argument(
            "Canonical event time must be an RFC3339 UTC timestamp ending in Z: " + value);
    }

    std::string fraction;
    if (value.size() == 20) {
        fraction = "000";
    } else {
        if (value[19] != '.' || value.size() < 22
            || !allDigits(std::string_view(value).substr(20, value.size() - 21))) {
            throw std::invalid_argument(
                "Canonical event time has an invalid fractional second: " + value);
        }
        fraction = value.substr(20, value.size() - 21);
        fraction.resize(3, '0');
        fraction.resize(3);
    }

    const int year_value = decimal(std::string_view(value).substr(0, 4));
    const int month_value = decimal(std::string_view(value).substr(5, 2));
    const int day_value = decimal(std::string_view(value).substr(8, 2));
    const int hour_value = decimal(std::string_view(value).substr(11, 2));
    const int minute_value = decimal(std::string_view(value).substr(14, 2));
    const int second_value = decimal(std::string_view(value).substr(17, 2));
    const std::chrono::year_month_day date{
        std::chrono::year(year_value),
        std::chrono::month(static_cast<unsigned>(month_value)),
        std::chrono::day(static_cast<unsigned>(day_value)),
    };
    if (!date.ok() || hour_value > 23 || minute_value > 59 || second_value > 59) {
        throw std::invalid_argument(
            "Canonical event time is outside the supported UTC calendar range: " + value);
    }

    return {
        value.substr(0, 10),
        value.substr(11, 8) + "." + fraction,
        value.substr(0, 4) + value.substr(5, 2) + value.substr(8, 2) + "_"
            + value.substr(11, 2) + value.substr(14, 2) + value.substr(17, 2)
            + "_" + fraction,
    };
}

std::string csvEscape(std::string_view value) {
    if (value.find_first_of(",\"\r\n") == std::string::npos) return std::string(value);
    std::string escaped{"\""};
    for (const char character : value) {
        if (character == '"') escaped += "\"\"";
        else escaped += character;
    }
    return escaped + '"';
}

std::string safeFilePart(std::string value) {
    for (char& character : value) {
        const bool valid = (character >= 'a' && character <= 'z')
            || (character >= 'A' && character <= 'Z')
            || (character >= '0' && character <= '9') || character == '-' || character == '_';
        if (!valid) character = '_';
    }
    return value.empty() ? "UNKNOWN" : value;
}

std::string operatorEventType(const std::string& canonical_type) {
    if (canonical_type == "com.cuajone.safety.ppe.violation.v2") return "INCUMPLIMIENTO_EPP";
    if (canonical_type == "com.cuajone.safety.fall.possible.v2") return "POSIBLE_CAIDA";
    throw std::invalid_argument(
        "Unsupported canonical event type for operator evidence: " + canonical_type);
}

std::string missingItems(const std::optional<PpeEvaluation>& evaluation) {
    if (!evaluation || !evaluation->evaluated) return {};
    std::string result;
    for (const auto& item : evaluation->items) {
        if (item.present) continue;
        if (!result.empty()) result += ";";
        result += ppeItemLabel(item.item);
    }
    return result;
}

std::string formatConfidence(float confidence) {
    if (!std::isfinite(confidence)) {
        throw std::invalid_argument("Canonical event confidence must be finite");
    }
    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << std::fixed << std::setprecision(3) << confidence;
    std::string value = output.str();
    while (value.size() > 2 && value.back() == '0') value.pop_back();
    if (value.back() == '.') value.push_back('0');
    return value;
}

std::string pathUtf8(const std::filesystem::path& path) {
    const auto value = path.u8string();
    return {value.begin(), value.end()};
}

void validateExistingCsv(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) return;
    if (!std::filesystem::is_regular_file(path)) {
        throw std::runtime_error(
            "Operator event CSV path is not a regular file: " + path.string());
    }
    if (std::filesystem::file_size(path) == 0) return;
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Cannot validate operator event CSV: " + path.string());
    }
    std::string header;
    std::getline(input, header);
    if (!header.empty() && header.back() == '\r') header.pop_back();
    if (header.starts_with("\xEF\xBB\xBF")) header.erase(0, 3);
    if (header != kOperatorCsvHeader) {
        throw std::runtime_error(
            "Existing operator event CSV has an incompatible header: " + path.string());
    }
}

#ifdef _WIN32
class UniqueHandle {
public:
    explicit UniqueHandle(HANDLE value) noexcept : value_(value) {}
    ~UniqueHandle() { if (value_ != INVALID_HANDLE_VALUE) CloseHandle(value_); }
    UniqueHandle(const UniqueHandle&) = delete;
    UniqueHandle& operator=(const UniqueHandle&) = delete;
    [[nodiscard]] HANDLE get() const noexcept { return value_; }

private:
    HANDLE value_;
};

std::string windowsErrorMessage(
    const std::string& action,
    const std::filesystem::path& path) {
    return action + ": " + path.string() + " (Windows error "
        + std::to_string(GetLastError()) + ")";
}

void flushFile(const std::filesystem::path& path) {
    UniqueHandle file(CreateFileW(
        path.c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL, nullptr));
    if (file.get() == INVALID_HANDLE_VALUE) {
        throw std::runtime_error(
            windowsErrorMessage("Cannot open evidence file for durable flush", path));
    }
    if (!FlushFileBuffers(file.get())) {
        throw std::runtime_error(
            windowsErrorMessage("Cannot durably flush evidence file", path));
    }
}

void appendDurably(const std::filesystem::path& path, const std::string& row) {
    UniqueHandle file(CreateFileW(
        path.c_str(), FILE_APPEND_DATA, FILE_SHARE_READ, nullptr, OPEN_ALWAYS,
        FILE_ATTRIBUTE_NORMAL, nullptr));
    if (file.get() == INVALID_HANDLE_VALUE) {
        throw std::runtime_error(
            windowsErrorMessage("Cannot append operator event CSV", path));
    }
    LARGE_INTEGER size{};
    if (!GetFileSizeEx(file.get(), &size)) {
        throw std::runtime_error(
            windowsErrorMessage("Cannot inspect operator event CSV", path));
    }
    const std::string bytes = size.QuadPart == 0
        ? std::string(kOperatorCsvHeader) + "\n" + row
        : row;
    if (bytes.size() > std::numeric_limits<DWORD>::max()) {
        throw std::runtime_error("Operator event CSV row exceeds the Windows write limit");
    }
    DWORD written{};
    if (!WriteFile(
            file.get(), bytes.data(), static_cast<DWORD>(bytes.size()), &written, nullptr)
        || written != bytes.size()) {
        throw std::runtime_error(
            windowsErrorMessage("Operator event CSV write failed", path));
    }
    if (!FlushFileBuffers(file.get())) {
        throw std::runtime_error(
            windowsErrorMessage("Operator event CSV durable flush failed", path));
    }
}
#else
void flushFile(const std::filesystem::path&) {}

void appendDurably(const std::filesystem::path& path, const std::string& row) {
    const bool needs_header = !std::filesystem::exists(path)
        || std::filesystem::file_size(path) == 0;
    std::ofstream output(path, std::ios::binary | std::ios::app);
    if (!output) {
        throw std::runtime_error("Cannot append operator event CSV: " + path.string());
    }
    if (needs_header) output << kOperatorCsvHeader << '\n';
    output << row;
    output.flush();
    if (!output) {
        throw std::runtime_error("Operator event CSV write failed: " + path.string());
    }
}
#endif

std::string csvRow(const EvidenceRecord& record) {
    const std::vector<std::string> fields{
        std::string(kOperatorEvidenceContractVersion),
        record.event_id,
        record.camera,
        record.date,
        record.time,
        record.event_type,
        record.ppe_states[0], record.ppe_states[1], record.ppe_states[2],
        record.ppe_states[3], record.ppe_states[4], record.ppe_states[5],
        record.ppe_states[6], record.missing_items,
        record.ppe_status,
        formatConfidence(record.ppe_ratios[0]), formatConfidence(record.ppe_ratios[1]),
        formatConfidence(record.ppe_ratios[2]), formatConfidence(record.ppe_ratios[3]),
        formatConfidence(record.ppe_ratios[4]), formatConfidence(record.ppe_ratios[5]),
        formatConfidence(record.ppe_ratios[6]), formatConfidence(record.ppe_confidences[0]),
        formatConfidence(record.ppe_confidences[1]), formatConfidence(record.ppe_confidences[2]),
        formatConfidence(record.ppe_confidences[3]), formatConfidence(record.ppe_confidences[4]),
        formatConfidence(record.ppe_confidences[5]), formatConfidence(record.ppe_confidences[6]),
        formatConfidence(record.confidence),
        std::to_string(record.track_id),
        record.review_status,
        record.human_identification,
        record.review_notes,
        pathUtf8(record.image_path),
    };
    std::string row;
    for (std::size_t index = 0; index < fields.size(); ++index) {
        if (index != 0) row.push_back(',');
        row += csvEscape(fields[index]);
    }
    return row + '\n';
}

}  // namespace

void validateWritableOutput(const std::filesystem::path& output) {
    std::filesystem::create_directories(output);
    const auto probe = output / ".cuajone-write-probe.tmp";
    {
        std::ofstream stream(probe, std::ios::binary | std::ios::trunc);
        if (!stream) {
            throw std::runtime_error("Output directory is not writable: " + output.string());
        }
        stream << "probe";
        if (!stream) {
            throw std::runtime_error("Output write probe failed: " + output.string());
        }
    }
    std::error_code error;
    std::filesystem::remove(probe, error);
    if (error) {
        throw std::runtime_error(
            "Output write probe could not be removed: " + error.message());
    }
}

EvidenceWriter::EvidenceWriter(std::filesystem::path output)
    : output_(std::move(output)), images_(output_ / "Evidencias"),
      csv_(output_ / "Reporte_Eventos_Seguridad_v2.csv") {
    validateWritableOutput(output_);
    std::filesystem::create_directories(images_);
    validateExistingCsv(csv_);
}

EvidenceRecord EvidenceWriter::append(
    const cv::Mat& annotated_frame,
    const std::string& source_label,
    const CanonicalEvent& event) {
    if (event.id.empty()) {
        throw std::invalid_argument("Canonical event ID is required for operator evidence");
    }
    const ParsedTimestamp timestamp = parseUtcTimestamp(event.time);
    const std::string event_type = operatorEventType(event.type);
    const std::string id_suffix = event.id.substr(event.id.size() > 8 ? event.id.size() - 8 : 0);
    const std::string filename = safeFilePart(source_label) + "_" + safeFilePart(event_type)
        + "_" + timestamp.filename + "_" + safeFilePart(id_suffix) + ".jpg";
    const auto image_path = images_ / filename;
    const auto temporary_path = images_
        / (filename + ".tmp-" + std::to_string(++sequence_) + ".jpg");
    try {
        if (annotated_frame.empty()
            || !cv::imwrite(temporary_path.string(), annotated_frame)) {
            throw std::runtime_error(
                "OpenCV could not write evidence image: " + image_path.string());
        }
        flushFile(temporary_path);
        std::error_code rename_error;
        std::filesystem::rename(temporary_path, image_path, rename_error);
        if (rename_error) {
            throw std::runtime_error(
                "Could not finalize evidence image: " + rename_error.message());
        }
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temporary_path, ignored);
        throw;
    }

    EvidenceRecord record{
        event.id,
        source_label,
        timestamp.date,
        timestamp.time,
        event_type,
        {},
        {},
        {},
        missingItems(event.ppe),
        event.status == "Evaluating PPE" || event.status == "Evaluando EPP"
            ? "En evaluaci\xC3\xB3n" : event.status,
        event.confidence,
        event.track_id,
        "PENDIENTE",
        {},
        {},
        image_path,
    };
    for (std::size_t index = 0; index < kPpeItemCount; ++index) {
        record.ppe_states[index] = "N/D";
    }
    if (event.ppe) {
        for (const auto& item : event.ppe->items) {
            const auto index = static_cast<std::size_t>(item.item);
            record.ppe_states[index] = event.ppe->evaluated ? (item.present ? "SI" : "NO") : "N/D";
            record.ppe_ratios[index] = item.ratio;
            record.ppe_confidences[index] = item.confidence;
        }
    }
    appendDurably(csv_, csvRow(record));
    return record;
}

}  // namespace cuajone
