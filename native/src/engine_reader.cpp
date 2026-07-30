// SPDX-License-Identifier: AGPL-3.0-only

#include "cuajone/engine_reader.hpp"
#include "cuajone/resource_limits.hpp"

#include <algorithm>
#include <charconv>
#include <cstring>
#include <fstream>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <variant>

namespace cuajone {
namespace {

constexpr std::uint32_t kMaximumMetadataBytes = 16U * 1024U * 1024U;

struct JsonValue {
    using Object = std::map<std::string, JsonValue>;
    using Array = std::vector<JsonValue>;
    std::variant<std::nullptr_t, bool, double, std::string, Object, Array> value;
};

class JsonParser {
public:
    explicit JsonParser(std::string_view input) : input_(input) {}

    JsonValue parse() {
        JsonValue result = parseValue();
        skipWhitespace();
        if (position_ != input_.size()) {
            fail("unexpected trailing data");
        }
        return result;
    }

private:
    [[noreturn]] void fail(std::string_view message) const {
        throw std::runtime_error(
            "Invalid Ultralytics engine metadata JSON at byte " +
            std::to_string(position_) + ": " + std::string(message));
    }

    void skipWhitespace() {
        while (position_ < input_.size()) {
            const char character = input_[position_];
            if (character != ' ' && character != '\n' && character != '\r' && character != '\t') {
                break;
            }
            ++position_;
        }
    }

    bool consume(char expected) {
        skipWhitespace();
        if (position_ < input_.size() && input_[position_] == expected) {
            ++position_;
            return true;
        }
        return false;
    }

    JsonValue parseValue() {
        skipWhitespace();
        if (position_ >= input_.size()) {
            fail("expected a value");
        }
        switch (input_[position_]) {
            case '{': return JsonValue{parseObject()};
            case '[': return JsonValue{parseArray()};
            case '"': return JsonValue{parseString()};
            case 't': parseLiteral("true"); return JsonValue{true};
            case 'f': parseLiteral("false"); return JsonValue{false};
            case 'n': parseLiteral("null"); return JsonValue{nullptr};
            default: return JsonValue{parseNumber()};
        }
    }

    JsonValue::Object parseObject() {
        if (!consume('{')) {
            fail("expected object");
        }
        JsonValue::Object object;
        if (consume('}')) {
            return object;
        }
        while (true) {
            skipWhitespace();
            if (position_ >= input_.size() || input_[position_] != '"') {
                fail("expected an object key");
            }
            std::string key = parseString();
            if (!consume(':')) {
                fail("expected ':' after object key");
            }
            if (!object.emplace(std::move(key), parseValue()).second) {
                fail("duplicate object key");
            }
            if (consume('}')) {
                break;
            }
            if (!consume(',')) {
                fail("expected ',' between object entries");
            }
        }
        return object;
    }

    JsonValue::Array parseArray() {
        if (!consume('[')) {
            fail("expected array");
        }
        JsonValue::Array array;
        if (consume(']')) {
            return array;
        }
        while (true) {
            array.push_back(parseValue());
            if (consume(']')) {
                break;
            }
            if (!consume(',')) {
                fail("expected ',' between array entries");
            }
        }
        return array;
    }

    std::string parseString() {
        if (input_[position_++] != '"') {
            fail("expected string");
        }
        std::string result;
        while (position_ < input_.size()) {
            const unsigned char character = static_cast<unsigned char>(input_[position_++]);
            if (character == '"') {
                return result;
            }
            if (character < 0x20U) {
                fail("control character in string");
            }
            if (character != '\\') {
                result.push_back(static_cast<char>(character));
                continue;
            }
            if (position_ >= input_.size()) {
                fail("unterminated escape sequence");
            }
            const char escaped = input_[position_++];
            switch (escaped) {
                case '"': result.push_back('"'); break;
                case '\\': result.push_back('\\'); break;
                case '/': result.push_back('/'); break;
                case 'b': result.push_back('\b'); break;
                case 'f': result.push_back('\f'); break;
                case 'n': result.push_back('\n'); break;
                case 'r': result.push_back('\r'); break;
                case 't': result.push_back('\t'); break;
                case 'u': parseUnicodeEscape(result); break;
                default: fail("unsupported escape sequence");
            }
        }
        fail("unterminated string");
    }

    void parseUnicodeEscape(std::string& output) {
        const unsigned int first = parseHexCodeUnit();
        unsigned int codepoint = first;
        if (first >= 0xD800U && first <= 0xDBFFU) {
            if (position_ + 2 > input_.size() || input_[position_] != '\\'
                || input_[position_ + 1] != 'u') {
                fail("high Unicode surrogate must be followed by a low surrogate");
            }
            position_ += 2;
            const unsigned int second = parseHexCodeUnit();
            if (second < 0xDC00U || second > 0xDFFFU) {
                fail("high Unicode surrogate must be followed by a low surrogate");
            }
            codepoint = 0x10000U + ((first - 0xD800U) << 10U) + (second - 0xDC00U);
        } else if (first >= 0xDC00U && first <= 0xDFFFU) {
            fail("lone low Unicode surrogate");
        }
        appendUtf8(output, codepoint);
    }

    unsigned int parseHexCodeUnit() {
        if (position_ + 4 > input_.size()) {
            fail("incomplete Unicode escape");
        }
        unsigned int code_unit = 0;
        for (int index = 0; index < 4; ++index) {
            const char digit = input_[position_++];
            code_unit <<= 4U;
            if (digit >= '0' && digit <= '9') code_unit += static_cast<unsigned int>(digit - '0');
            else if (digit >= 'a' && digit <= 'f') code_unit += static_cast<unsigned int>(digit - 'a' + 10);
            else if (digit >= 'A' && digit <= 'F') code_unit += static_cast<unsigned int>(digit - 'A' + 10);
            else fail("invalid Unicode escape");
        }
        return code_unit;
    }

    static void appendUtf8(std::string& output, unsigned int codepoint) {
        if (codepoint <= 0x7FU) {
            output.push_back(static_cast<char>(codepoint));
        } else if (codepoint <= 0x7FFU) {
            output.push_back(static_cast<char>(0xC0U | (codepoint >> 6U)));
            output.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
        } else if (codepoint <= 0xFFFFU) {
            output.push_back(static_cast<char>(0xE0U | (codepoint >> 12U)));
            output.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
            output.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
        } else {
            output.push_back(static_cast<char>(0xF0U | (codepoint >> 18U)));
            output.push_back(static_cast<char>(0x80U | ((codepoint >> 12U) & 0x3FU)));
            output.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
            output.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
        }
    }

    double parseNumber() {
        const std::size_t start = position_;
        if (input_[position_] == '-') ++position_;
        if (position_ >= input_.size()) fail("expected a number");
        if (input_[position_] == '0') {
            ++position_;
            if (position_ < input_.size() && input_[position_] >= '0' && input_[position_] <= '9') {
                fail("leading zero in number");
            }
        } else if (input_[position_] >= '1' && input_[position_] <= '9') {
            do {
                ++position_;
            } while (position_ < input_.size() && input_[position_] >= '0' && input_[position_] <= '9');
        } else {
            fail("expected a number");
        }
        if (position_ < input_.size() && input_[position_] == '.') {
            ++position_;
            const std::size_t fraction_start = position_;
            while (position_ < input_.size() && input_[position_] >= '0' && input_[position_] <= '9') ++position_;
            if (fraction_start == position_) fail("fraction requires at least one digit");
        }
        if (position_ < input_.size() && (input_[position_] == 'e' || input_[position_] == 'E')) {
            ++position_;
            if (position_ < input_.size() && (input_[position_] == '+' || input_[position_] == '-')) ++position_;
            const std::size_t exponent_start = position_;
            while (position_ < input_.size() && input_[position_] >= '0' && input_[position_] <= '9') ++position_;
            if (exponent_start == position_) fail("exponent requires at least one digit");
        }
        double result{};
        const auto parsed = std::from_chars(input_.data() + start, input_.data() + position_, result);
        if (parsed.ec != std::errc{} || parsed.ptr != input_.data() + position_
            || !std::isfinite(result)) {
            fail("invalid number");
        }
        return result;
    }

    void parseLiteral(std::string_view literal) {
        if (input_.substr(position_, literal.size()) != literal) {
            fail("invalid literal");
        }
        position_ += literal.size();
    }

    std::string_view input_;
    std::size_t position_{};
};

const JsonValue::Object& asObject(const JsonValue& value, std::string_view field) {
    const auto* object = std::get_if<JsonValue::Object>(&value.value);
    if (object == nullptr) {
        throw std::runtime_error("Engine metadata field '" + std::string(field) + "' must be an object");
    }
    return *object;
}

int positiveInteger(const JsonValue& value, std::string_view field) {
    const auto* number = std::get_if<double>(&value.value);
    if (number == nullptr || *number <= 0.0 || *number > static_cast<double>(std::numeric_limits<int>::max()) || *number != static_cast<int>(*number)) {
        throw std::runtime_error("Engine metadata field '" + std::string(field) + "' must contain positive integers");
    }
    return static_cast<int>(*number);
}

std::optional<std::array<int, 2>> parsePair(
    const JsonValue::Object& root,
    std::string_view key,
    bool allow_scalar) {
    const auto iterator = root.find(std::string(key));
    if (iterator == root.end()) {
        return std::nullopt;
    }
    if (std::holds_alternative<std::nullptr_t>(iterator->second.value)) {
        return std::nullopt;
    }
    if (allow_scalar) {
        if (std::holds_alternative<double>(iterator->second.value)) {
            const int value = positiveInteger(iterator->second, key);
            return std::array{value, value};
        }
    }
    const auto* array = std::get_if<JsonValue::Array>(&iterator->second.value);
    if (array == nullptr || array->size() != 2) {
        throw std::runtime_error("Engine metadata field '" + std::string(key) + "' must have two integers");
    }
    return std::array{
        positiveInteger((*array)[0], key),
        positiveInteger((*array)[1], key),
    };
}

EngineMetadata extractMetadata(std::string_view json) {
    const JsonValue parsed = JsonParser(json).parse();
    const auto& root = asObject(parsed, "root");
    EngineMetadata metadata;

    if (const auto task = root.find("task"); task != root.end()) {
        const auto* value = std::get_if<std::string>(&task->second.value);
        if (value == nullptr || value->empty()) {
            throw std::runtime_error("Engine metadata field 'task' must be a non-empty string");
        }
        metadata.task = *value;
    }
    metadata.image_size = parsePair(root, "imgsz", true);
    metadata.keypoint_shape = parsePair(root, "kpt_shape", false);

    if (const auto names = root.find("names"); names != root.end()) {
        if (std::holds_alternative<std::nullptr_t>(names->second.value)) {
            return metadata;
        }
        if (const auto* array = std::get_if<JsonValue::Array>(&names->second.value)) {
            for (std::size_t index = 0; index < array->size(); ++index) {
                const auto* name = std::get_if<std::string>(&(*array)[index].value);
                if (name == nullptr || name->empty()) {
                    throw std::runtime_error("Engine metadata names array must contain non-empty strings");
                }
                metadata.names.emplace(static_cast<int>(index), *name);
            }
        } else {
            const auto& object = asObject(names->second, "names");
            for (const auto& [id_text, value] : object) {
                int id{};
                const auto converted = std::from_chars(id_text.data(), id_text.data() + id_text.size(), id);
                const auto* name = std::get_if<std::string>(&value.value);
                if (converted.ec != std::errc{} || converted.ptr != id_text.data() + id_text.size() || id < 0 || name == nullptr || name->empty()) {
                    throw std::runtime_error("Engine metadata names object must map non-negative integer keys to strings");
                }
                if (!metadata.names.emplace(id, *name).second) {
                    throw std::runtime_error("Engine metadata names object contains duplicate numeric class IDs");
                }
            }
        }
    }
    return metadata;
}

std::uint32_t readLittleEndianLength(std::span<const std::byte> bytes) {
    return static_cast<std::uint32_t>(std::to_integer<unsigned char>(bytes[0]))
        | (static_cast<std::uint32_t>(std::to_integer<unsigned char>(bytes[1])) << 8U)
        | (static_cast<std::uint32_t>(std::to_integer<unsigned char>(bytes[2])) << 16U)
        | (static_cast<std::uint32_t>(std::to_integer<unsigned char>(bytes[3])) << 24U);
}

}  // namespace

EngineFile EngineFile::read(const std::filesystem::path& path) {
    if (path.extension() != ".engine") {
        throw std::runtime_error("TensorRT artifact path must use the .engine extension: " + path.string());
    }
    const auto status = std::filesystem::symlink_status(path);
    if (!std::filesystem::is_regular_file(status) || std::filesystem::is_symlink(status)) {
        throw std::runtime_error("TensorRT engine must be a regular non-symlink file: " + path.string());
    }
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        throw std::runtime_error("Cannot open TensorRT engine: " + path.string());
    }
    const std::streamsize file_size = stream.tellg();
    if (file_size <= 0) {
        throw std::runtime_error("TensorRT engine is empty: " + path.string());
    }
    if (static_cast<std::uintmax_t>(file_size) > resource_limits::kMaximumTensorRtEngineBytes) {
        throw std::runtime_error("TensorRT engine exceeds the 1 GiB artifact limit: " + path.string());
    }
    stream.seekg(0);

    EngineFile result;
    result.bytes_.resize(static_cast<std::size_t>(file_size));
    if (!stream.read(reinterpret_cast<char*>(result.bytes_.data()), file_size)) {
        throw std::runtime_error("Cannot read complete TensorRT engine: " + path.string());
    }

    if (result.bytes_.size() >= 5) {
        const std::uint32_t metadata_length = readLittleEndianLength(result.bytes_);
        const std::size_t plan_offset = 4ULL + metadata_length;
        std::size_t json_start_offset = 4;
        const std::size_t available_metadata_end = std::min(plan_offset, result.bytes_.size());
        while (json_start_offset < available_metadata_end) {
            const char character = static_cast<char>(result.bytes_[json_start_offset]);
            if (character != ' ' && character != '\n' && character != '\r' && character != '\t') break;
            ++json_start_offset;
        }
        const bool looks_like_json = json_start_offset < available_metadata_end
            && (result.bytes_[json_start_offset] == std::byte{'{'}
                || result.bytes_[json_start_offset] == std::byte{'['});
        if (metadata_length > 0 && looks_like_json) {
            if (metadata_length > kMaximumMetadataBytes) {
                throw std::runtime_error("Ultralytics engine metadata prefix exceeds the 16 MiB limit");
            }
            if (plan_offset >= result.bytes_.size()) {
                throw std::runtime_error("Ultralytics engine metadata prefix is truncated or has no TensorRT plan");
            }
            if (result.bytes_[json_start_offset] == std::byte{'['}) {
                throw std::runtime_error("Ultralytics engine metadata prefix must contain a JSON object");
            }
            const auto* json_start = reinterpret_cast<const char*>(result.bytes_.data() + 4);
            result.metadata_ = extractMetadata(std::string_view(json_start, metadata_length));
            result.plan_offset_ = plan_offset;
            result.has_metadata_prefix_ = true;
        }
    }

    if (result.plan().empty()) {
        throw std::runtime_error("TensorRT engine contains no serialized plan: " + path.string());
    }
    return result;
}

std::span<const std::byte> EngineFile::plan() const noexcept {
    return std::span(bytes_).subspan(plan_offset_);
}

const EngineMetadata& EngineFile::metadata() const noexcept {
    return metadata_;
}

bool EngineFile::hasMetadataPrefix() const noexcept {
    return has_metadata_prefix_;
}

}  // namespace cuajone
