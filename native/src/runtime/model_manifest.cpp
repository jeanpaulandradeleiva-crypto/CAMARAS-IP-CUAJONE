// SPDX-License-Identifier: AGPL-3.0-only

#include "cuajone/model_manifest.hpp"
#include "cuajone/resource_limits.hpp"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <bcrypt.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <variant>

namespace cuajone {
namespace {

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
        if (position_ != input_.size()) fail("unexpected trailing data");
        return result;
    }

private:
    [[noreturn]] void fail(std::string_view message) const {
        throw std::runtime_error(
            "Invalid ONNX manifest JSON at byte " + std::to_string(position_) + ": "
            + std::string(message));
    }

    void skipWhitespace() {
        while (position_ < input_.size()) {
            const char character = input_[position_];
            if (character != ' ' && character != '\n' && character != '\r' && character != '\t') break;
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
        if (position_ >= input_.size()) fail("expected a value");
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
        if (!consume('{')) fail("expected object");
        JsonValue::Object object;
        if (consume('}')) return object;
        while (true) {
            skipWhitespace();
            if (position_ >= input_.size() || input_[position_] != '"') fail("expected an object key");
            std::string key = parseString();
            if (!consume(':')) fail("expected ':' after object key");
            if (!object.emplace(std::move(key), parseValue()).second) fail("duplicate object key");
            if (consume('}')) break;
            if (!consume(',')) fail("expected ',' between object entries");
        }
        return object;
    }

    JsonValue::Array parseArray() {
        if (!consume('[')) fail("expected array");
        JsonValue::Array array;
        if (consume(']')) return array;
        while (true) {
            array.push_back(parseValue());
            if (consume(']')) break;
            if (!consume(',')) fail("expected ',' between array entries");
        }
        return array;
    }

    std::string parseString() {
        if (input_[position_++] != '"') fail("expected string");
        std::string result;
        while (position_ < input_.size()) {
            const unsigned char character = static_cast<unsigned char>(input_[position_++]);
            if (character == '"') return result;
            if (character < 0x20U) fail("control character in string");
            if (character != '\\') {
                result.push_back(static_cast<char>(character));
                continue;
            }
            if (position_ >= input_.size()) fail("unterminated escape sequence");
            switch (input_[position_++]) {
                case '"': result.push_back('"'); break;
                case '\\': result.push_back('\\'); break;
                case '/': result.push_back('/'); break;
                case 'b': result.push_back('\b'); break;
                case 'f': result.push_back('\f'); break;
                case 'n': result.push_back('\n'); break;
                case 'r': result.push_back('\r'); break;
                case 't': result.push_back('\t'); break;
                case 'u': appendUnicode(result); break;
                default: fail("unsupported escape sequence");
            }
        }
        fail("unterminated string");
    }

    unsigned int hexCodeUnit() {
        if (position_ + 4 > input_.size()) fail("incomplete Unicode escape");
        unsigned int result = 0;
        for (int index = 0; index < 4; ++index) {
            const char digit = input_[position_++];
            result <<= 4U;
            if (digit >= '0' && digit <= '9') result += static_cast<unsigned int>(digit - '0');
            else if (digit >= 'a' && digit <= 'f') result += static_cast<unsigned int>(digit - 'a' + 10);
            else if (digit >= 'A' && digit <= 'F') result += static_cast<unsigned int>(digit - 'A' + 10);
            else fail("invalid Unicode escape");
        }
        return result;
    }

    void appendUnicode(std::string& output) {
        const unsigned int first = hexCodeUnit();
        unsigned int codepoint = first;
        if (first >= 0xD800U && first <= 0xDBFFU) {
            if (position_ + 2 > input_.size() || input_[position_] != '\\' || input_[position_ + 1] != 'u') {
                fail("high Unicode surrogate requires a low surrogate");
            }
            position_ += 2;
            const unsigned int second = hexCodeUnit();
            if (second < 0xDC00U || second > 0xDFFFU) fail("invalid low Unicode surrogate");
            codepoint = 0x10000U + ((first - 0xD800U) << 10U) + (second - 0xDC00U);
        } else if (first >= 0xDC00U && first <= 0xDFFFU) {
            fail("lone low Unicode surrogate");
        }
        if (codepoint <= 0x7FU) output.push_back(static_cast<char>(codepoint));
        else if (codepoint <= 0x7FFU) {
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
            while (position_ < input_.size() && input_[position_] >= '0' && input_[position_] <= '9') ++position_;
        } else {
            fail("expected a number");
        }
        if (position_ < input_.size() && input_[position_] == '.') {
            ++position_;
            const std::size_t fraction = position_;
            while (position_ < input_.size() && input_[position_] >= '0' && input_[position_] <= '9') ++position_;
            if (fraction == position_) fail("fraction requires a digit");
        }
        if (position_ < input_.size() && (input_[position_] == 'e' || input_[position_] == 'E')) {
            ++position_;
            if (position_ < input_.size() && (input_[position_] == '+' || input_[position_] == '-')) ++position_;
            const std::size_t exponent = position_;
            while (position_ < input_.size() && input_[position_] >= '0' && input_[position_] <= '9') ++position_;
            if (exponent == position_) fail("exponent requires a digit");
        }
        double result{};
        const auto parsed = std::from_chars(input_.data() + start, input_.data() + position_, result);
        if (parsed.ec != std::errc{} || parsed.ptr != input_.data() + position_ || !std::isfinite(result)) {
            fail("invalid number");
        }
        return result;
    }

    void parseLiteral(std::string_view literal) {
        if (input_.substr(position_, literal.size()) != literal) fail("invalid literal");
        position_ += literal.size();
    }

    std::string_view input_;
    std::size_t position_{};
};

const JsonValue::Object& objectValue(const JsonValue& value, std::string_view name) {
    const auto* object = std::get_if<JsonValue::Object>(&value.value);
    if (object == nullptr) throw std::runtime_error(std::string(name) + " must be a JSON object");
    return *object;
}

const JsonValue::Array& arrayValue(const JsonValue& value, std::string_view name) {
    const auto* array = std::get_if<JsonValue::Array>(&value.value);
    if (array == nullptr) throw std::runtime_error(std::string(name) + " must be a JSON array");
    return *array;
}

const JsonValue& required(const JsonValue::Object& object, std::string_view key) {
    const auto iterator = object.find(std::string(key));
    if (iterator == object.end()) throw std::runtime_error("ONNX manifest is missing '" + std::string(key) + "'");
    return iterator->second;
}

void exactKeys(
    const JsonValue::Object& object,
    std::initializer_list<std::string_view> expected,
    std::string_view name) {
    if (object.size() != expected.size()) {
        throw std::runtime_error(std::string(name) + " contains missing or unsupported fields");
    }
    for (const auto key : expected) {
        if (!object.contains(std::string(key))) {
            throw std::runtime_error(std::string(name) + " is missing '" + std::string(key) + "'");
        }
    }
}

std::string stringValue(const JsonValue& value, std::string_view name, std::size_t maximum = 1024) {
    const auto* text = std::get_if<std::string>(&value.value);
    if (text == nullptr || text->empty() || text->size() > maximum) {
        throw std::runtime_error(std::string(name) + " must be a bounded non-empty string");
    }
    return *text;
}

std::size_t integerValue(const JsonValue& value, std::string_view name, std::size_t maximum) {
    const auto* number = std::get_if<double>(&value.value);
    if (number == nullptr || *number < 0.0 || *number > static_cast<double>(maximum)
        || *number != std::floor(*number)) {
        throw std::runtime_error(std::string(name) + " must be an integer in the supported range");
    }
    return static_cast<std::size_t>(*number);
}

bool boolValue(const JsonValue& value, std::string_view name) {
    const auto* result = std::get_if<bool>(&value.value);
    if (result == nullptr) throw std::runtime_error(std::string(name) + " must be boolean");
    return *result;
}

TensorContract tensorContract(const JsonValue& value, std::string_view name, std::size_t maximum_elements) {
    const auto& object = objectValue(value, name);
    exactKeys(object, {"name", "element_type", "shape"}, name);
    if (stringValue(required(object, "element_type"), "tensor element_type", 16) != "float32") {
        throw std::runtime_error(std::string(name) + " element_type must be float32");
    }
    TensorContract result;
    result.name = stringValue(required(object, "name"), "tensor name", 256);
    for (const auto& dimension : arrayValue(required(object, "shape"), "tensor shape")) {
        result.shape.push_back(static_cast<std::int64_t>(integerValue(
            dimension, "tensor dimension",
            static_cast<std::size_t>(resource_limits::kMaximumTensorDimension))));
    }
    resource_limits::checkedVolume(result.shape, maximum_elements, name);
    return result;
}

OnnxModelManifest parseManifest(std::string_view json) {
    const JsonValue parsed = JsonParser(json).parse();
    const auto& root = objectValue(parsed, "ONNX manifest");
    const std::string role = stringValue(required(root, "role"), "role", 16);
    if (role == "ppe") {
        exactKeys(root, {
            "schema_version", "artifact_type", "role", "model_file", "model_sha256",
            "model_size_bytes", "external_data", "custom_operators", "input", "output",
            "provenance", "label_contract", "labels",
        }, "ONNX manifest");
    } else {
        exactKeys(root, {
        "schema_version", "artifact_type", "role", "model_file", "model_sha256",
        "model_size_bytes", "external_data", "custom_operators", "input", "output",
        "provenance",
        }, "ONNX manifest");
    }
    if (integerValue(required(root, "schema_version"), "schema_version", 1) != 1) {
        throw std::runtime_error("Unsupported ONNX manifest schema_version");
    }
    if (stringValue(required(root, "artifact_type"), "artifact_type", 16) != "onnx") {
        throw std::runtime_error("ONNX manifest artifact_type must be onnx");
    }
    OnnxModelManifest result;
    if (role == "ppe") result.role = ModelRole::Ppe;
    else if (role == "pose") result.role = ModelRole::Pose;
    else throw std::runtime_error("ONNX manifest role must be ppe or pose");
    result.model_file = stringValue(required(root, "model_file"), "model_file", 255);
    result.model_sha256 = stringValue(required(root, "model_sha256"), "model_sha256", 64);
    if (result.model_sha256.size() != 64 || std::any_of(
            result.model_sha256.begin(), result.model_sha256.end(), [](unsigned char character) {
                return !((character >= '0' && character <= '9') || (character >= 'a' && character <= 'f'));
            })) {
        throw std::runtime_error("ONNX manifest model_sha256 must be 64 lowercase hexadecimal characters");
    }
    result.model_size_bytes = integerValue(
        required(root, "model_size_bytes"), "model_size_bytes",
        resource_limits::kMaximumOnnxModelBytes);
    if (result.model_size_bytes == 0) throw std::runtime_error("ONNX model must not be empty");
    if (boolValue(required(root, "external_data"), "external_data")) {
        throw std::runtime_error("ONNX external data is prohibited");
    }
    if (boolValue(required(root, "custom_operators"), "custom_operators")) {
        throw std::runtime_error("ONNX custom operators are prohibited");
    }
    result.input = tensorContract(required(root, "input"), "ONNX input contract", resource_limits::kMaximumInputElements);
    result.output = tensorContract(required(root, "output"), "ONNX output contract", resource_limits::kMaximumOutputElements);
    if (result.input.shape.size() != 4 || result.input.shape[0] != 1 || result.input.shape[1] != 3
        || result.input.shape[2] > resource_limits::kMaximumImageDimension
        || result.input.shape[3] > resource_limits::kMaximumImageDimension) {
        throw std::runtime_error("ONNX input contract must be bounded batch-1, three-channel NCHW");
    }
    const auto& provenance = objectValue(required(root, "provenance"), "provenance");
    exactKeys(provenance, {"source_uri", "exporter", "license"}, "ONNX provenance");
    result.source_uri = stringValue(required(provenance, "source_uri"), "source_uri");
    if (!result.source_uri.starts_with("https://") && !result.source_uri.starts_with("urn:")) {
        throw std::runtime_error("ONNX provenance source_uri must use https:// or urn:");
    }
    result.exporter = stringValue(required(provenance, "exporter"), "exporter", 256);
    result.license = stringValue(required(provenance, "license"), "license", 256);
    if (result.role == ModelRole::Ppe) {
        result.label_contract = stringValue(required(root, "label_contract"), "label_contract", 64);
        for (const auto& label : arrayValue(required(root, "labels"), "PPE labels")) {
            result.labels.push_back(stringValue(label, "PPE label", 64));
        }
        if (result.label_contract != "always-all-seven-v2" || result.labels.size() != 8) {
            throw std::runtime_error("PPE manifest must bind the always-all-seven-v2 label contract");
        }
    }
    return result;
}

std::vector<std::byte> readBoundedFile(
    const std::filesystem::path& path,
    std::size_t maximum,
    std::string_view name) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("Cannot open " + std::string(name) + ": " + path.string());
    const std::streamoff end = stream.tellg();
    if (end <= 0 || static_cast<std::uintmax_t>(end) > maximum) {
        throw std::runtime_error(std::string(name) + " is empty or exceeds its byte limit");
    }
    stream.seekg(0);
    std::vector<std::byte> bytes(static_cast<std::size_t>(end));
    if (!stream.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()))) {
        throw std::runtime_error("Cannot read complete " + std::string(name) + ": " + path.string());
    }
    return bytes;
}

std::string calculateSha256(std::span<const std::byte> bytes) {
    BCRYPT_ALG_HANDLE algorithm = nullptr;
    BCRYPT_HASH_HANDLE hash = nullptr;
    if (BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0) < 0) {
        throw std::runtime_error("BCryptOpenAlgorithmProvider SHA-256 failed");
    }
    DWORD object_size{};
    DWORD hash_size{};
    DWORD copied{};
    std::vector<unsigned char> object;
    std::vector<unsigned char> digest;
    try {
        if (BCryptGetProperty(algorithm, BCRYPT_OBJECT_LENGTH, reinterpret_cast<PUCHAR>(&object_size),
                sizeof(object_size), &copied, 0) < 0
            || BCryptGetProperty(algorithm, BCRYPT_HASH_LENGTH, reinterpret_cast<PUCHAR>(&hash_size),
                sizeof(hash_size), &copied, 0) < 0) {
            throw std::runtime_error("BCryptGetProperty SHA-256 failed");
        }
        object.resize(object_size);
        digest.resize(hash_size);
        if (BCryptCreateHash(algorithm, &hash, object.data(), object_size, nullptr, 0, 0) < 0
            || BCryptHashData(hash, reinterpret_cast<PUCHAR>(const_cast<std::byte*>(bytes.data())),
                static_cast<ULONG>(bytes.size()), 0) < 0
            || BCryptFinishHash(hash, digest.data(), hash_size, 0) < 0) {
            throw std::runtime_error("BCrypt SHA-256 hashing failed");
        }
    } catch (...) {
        if (hash != nullptr) BCryptDestroyHash(hash);
        BCryptCloseAlgorithmProvider(algorithm, 0);
        throw;
    }
    BCryptDestroyHash(hash);
    BCryptCloseAlgorithmProvider(algorithm, 0);
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (const unsigned char byte : digest) output << std::setw(2) << static_cast<unsigned int>(byte);
    return output.str();
}

class ProtoReader {
public:
    explicit ProtoReader(std::span<const std::byte> bytes) : bytes_(bytes) {}

    bool done() const noexcept { return position_ == bytes_.size(); }

    std::pair<std::uint32_t, std::uint32_t> tag() {
        const std::uint64_t encoded = varint();
        const auto field = static_cast<std::uint32_t>(encoded >> 3U);
        const auto wire = static_cast<std::uint32_t>(encoded & 7U);
        if (field == 0 || (wire != 0 && wire != 1 && wire != 2 && wire != 5)) {
            throw std::runtime_error("ONNX protobuf contains an invalid field tag");
        }
        return {field, wire};
    }

    std::uint64_t varint() {
        std::uint64_t result = 0;
        for (unsigned int shift = 0; shift < 70; shift += 7) {
            if (position_ >= bytes_.size()) throw std::runtime_error("ONNX protobuf varint is truncated");
            const auto byte = std::to_integer<unsigned char>(bytes_[position_++]);
            if (shift == 63 && (byte & 0xFEU) != 0) throw std::runtime_error("ONNX protobuf varint overflows");
            result |= static_cast<std::uint64_t>(byte & 0x7FU) << shift;
            if ((byte & 0x80U) == 0) return result;
        }
        throw std::runtime_error("ONNX protobuf varint is too long");
    }

    std::span<const std::byte> bytes() {
        const std::uint64_t length = varint();
        if (length > bytes_.size() - position_) throw std::runtime_error("ONNX protobuf field is truncated");
        const auto result = bytes_.subspan(position_, static_cast<std::size_t>(length));
        position_ += static_cast<std::size_t>(length);
        return result;
    }

    void skip(std::uint32_t wire) {
        switch (wire) {
            case 0: static_cast<void>(varint()); return;
            case 1: advance(8); return;
            case 2: static_cast<void>(bytes()); return;
            case 5: advance(4); return;
            default: throw std::runtime_error("ONNX protobuf uses an unsupported wire type");
        }
    }

private:
    void advance(std::size_t count) {
        if (count > bytes_.size() - position_) throw std::runtime_error("ONNX protobuf field is truncated");
        position_ += count;
    }

    std::span<const std::byte> bytes_;
    std::size_t position_{};
};

class OnnxSecurityScanner {
public:
    void scan(std::span<const std::byte> bytes) { scanModel(bytes, 0); }

private:
    static void requireWire(std::uint32_t actual, std::uint32_t expected, std::string_view field) {
        if (actual != expected) throw std::runtime_error("ONNX protobuf " + std::string(field) + " has the wrong wire type");
    }

    void field() {
        if (++fields_ > 4'000'000) throw std::runtime_error("ONNX protobuf exceeds the field-count limit");
    }

    void depth(std::size_t value) {
        if (value > 32) throw std::runtime_error("ONNX protobuf exceeds the nesting limit");
    }

    static void standardDomain(std::span<const std::byte> bytes) {
        const std::string_view domain(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        if (!domain.empty() && domain != "ai.onnx") {
            throw std::runtime_error("ONNX custom operator domain is prohibited: " + std::string(domain));
        }
    }

    void scanModel(std::span<const std::byte> bytes, std::size_t level) {
        depth(level);
        ProtoReader reader(bytes);
        while (!reader.done()) {
            field();
            const auto [number, wire] = reader.tag();
            if (number == 7) { requireWire(wire, 2, "graph"); scanGraph(reader.bytes(), level + 1); }
            else if (number == 8) { requireWire(wire, 2, "opset_import"); scanOpset(reader.bytes()); }
            else if (number == 20) { requireWire(wire, 2, "training_info"); scanTrainingInfo(reader.bytes(), level + 1); }
            else if (number == 25) throw std::runtime_error("ONNX local function/custom operator definitions are prohibited");
            else reader.skip(wire);
        }
    }

    void scanGraph(std::span<const std::byte> bytes, std::size_t level) {
        depth(level);
        ProtoReader reader(bytes);
        while (!reader.done()) {
            field();
            const auto [number, wire] = reader.tag();
            if (number == 1) { requireWire(wire, 2, "node"); scanNode(reader.bytes(), level + 1); }
            else if (number == 5) { requireWire(wire, 2, "initializer"); scanTensor(reader.bytes()); }
            else if (number == 15) { requireWire(wire, 2, "sparse_initializer"); scanSparseTensor(reader.bytes()); }
            else reader.skip(wire);
        }
    }

    void scanNode(std::span<const std::byte> bytes, std::size_t level) {
        depth(level);
        ProtoReader reader(bytes);
        while (!reader.done()) {
            field();
            const auto [number, wire] = reader.tag();
            if (number == 5) { requireWire(wire, 2, "attribute"); scanAttribute(reader.bytes(), level + 1); }
            else if (number == 7) { requireWire(wire, 2, "node domain"); standardDomain(reader.bytes()); }
            else reader.skip(wire);
        }
    }

    void scanAttribute(std::span<const std::byte> bytes, std::size_t level) {
        depth(level);
        ProtoReader reader(bytes);
        while (!reader.done()) {
            field();
            const auto [number, wire] = reader.tag();
            if (number == 5 || number == 10) { requireWire(wire, 2, "tensor attribute"); scanTensor(reader.bytes()); }
            else if (number == 6 || number == 11) { requireWire(wire, 2, "graph attribute"); scanGraph(reader.bytes(), level + 1); }
            else if (number == 22 || number == 23) { requireWire(wire, 2, "sparse tensor attribute"); scanSparseTensor(reader.bytes()); }
            else reader.skip(wire);
        }
    }

    void scanTensor(std::span<const std::byte> bytes) {
        ProtoReader reader(bytes);
        while (!reader.done()) {
            field();
            const auto [number, wire] = reader.tag();
            if (number == 13) throw std::runtime_error("ONNX TensorProto external_data entries are prohibited");
            if (number == 14) {
                requireWire(wire, 0, "data_location");
                if (reader.varint() != 0) throw std::runtime_error("ONNX external tensor data_location is prohibited");
            } else {
                reader.skip(wire);
            }
        }
    }

    void scanSparseTensor(std::span<const std::byte> bytes) {
        ProtoReader reader(bytes);
        while (!reader.done()) {
            field();
            const auto [number, wire] = reader.tag();
            if (number == 1 || number == 2) { requireWire(wire, 2, "sparse tensor"); scanTensor(reader.bytes()); }
            else reader.skip(wire);
        }
    }

    void scanOpset(std::span<const std::byte> bytes) {
        ProtoReader reader(bytes);
        while (!reader.done()) {
            field();
            const auto [number, wire] = reader.tag();
            if (number == 1) { requireWire(wire, 2, "opset domain"); standardDomain(reader.bytes()); }
            else reader.skip(wire);
        }
    }

    void scanTrainingInfo(std::span<const std::byte> bytes, std::size_t level) {
        depth(level);
        ProtoReader reader(bytes);
        while (!reader.done()) {
            field();
            const auto [number, wire] = reader.tag();
            if (number == 1 || number == 2) { requireWire(wire, 2, "training graph"); scanGraph(reader.bytes(), level + 1); }
            else reader.skip(wire);
        }
    }

    std::size_t fields_{};
};

}  // namespace

std::filesystem::path onnxManifestPath(const std::filesystem::path& model_path) {
    std::filesystem::path result = model_path;
    result += ".manifest.json";
    return result;
}

std::string_view modelRoleName(ModelRole role) noexcept {
    return role == ModelRole::Ppe ? "ppe" : "pose";
}

std::string sha256Hex(std::span<const std::byte> bytes) {
    return calculateSha256(bytes);
}

VerifiedOnnxModel verifyOnnxModel(const std::filesystem::path& model_path, ModelRole expected_role) {
    if (model_path.extension() != ".onnx") {
        throw std::runtime_error("CPU model path must use the .onnx extension: " + model_path.string());
    }
    const std::filesystem::path manifest_path = onnxManifestPath(model_path);
    const auto model_status = std::filesystem::symlink_status(model_path);
    const auto manifest_status = std::filesystem::symlink_status(manifest_path);
    if (!std::filesystem::is_regular_file(model_status) || std::filesystem::is_symlink(model_status)) {
        throw std::runtime_error("ONNX model must be a regular non-symlink file: " + model_path.string());
    }
    if (!std::filesystem::is_regular_file(manifest_status) || std::filesystem::is_symlink(manifest_status)) {
        throw std::runtime_error("ONNX manifest must be a regular non-symlink file: " + manifest_path.string());
    }
    const auto manifest_bytes = readBoundedFile(
        manifest_path, resource_limits::kMaximumManifestBytes, "ONNX manifest");
    const std::string_view manifest_json(
        reinterpret_cast<const char*>(manifest_bytes.data()), manifest_bytes.size());
    OnnxModelManifest manifest = parseManifest(manifest_json);
    if (manifest.role != expected_role) {
        throw std::runtime_error(
            "ONNX manifest role is " + std::string(modelRoleName(manifest.role))
            + ", expected " + std::string(modelRoleName(expected_role)));
    }
    if (std::filesystem::path(manifest.model_file).filename().string() != manifest.model_file
        || manifest.model_file != model_path.filename().string()) {
        throw std::runtime_error("ONNX manifest model_file must exactly match the adjacent model filename");
    }
    std::vector<std::byte> model_bytes = readBoundedFile(
        model_path, resource_limits::kMaximumOnnxModelBytes, "ONNX model");
    if (model_bytes.size() != manifest.model_size_bytes) {
        throw std::runtime_error("ONNX model byte size does not match its manifest");
    }
    if (sha256Hex(model_bytes) != manifest.model_sha256) {
        throw std::runtime_error("ONNX model SHA-256 does not match its manifest");
    }
    validateOnnxModelSecurity(model_bytes);
    return {std::move(manifest), manifest_path, std::move(model_bytes)};
}

void validateOnnxModelSecurity(std::span<const std::byte> model_bytes) {
    if (model_bytes.empty() || model_bytes.size() > resource_limits::kMaximumOnnxModelBytes) {
        throw std::runtime_error("ONNX model is empty or exceeds its byte limit");
    }
    OnnxSecurityScanner{}.scan(model_bytes);
}

}  // namespace cuajone
