// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include "cuajone/model_manifest.hpp"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace cuajone::test {

using Bytes = std::vector<std::byte>;

inline void appendVarint(Bytes& output, std::uint64_t value) {
    while (value >= 0x80U) {
        output.push_back(static_cast<std::byte>((value & 0x7FU) | 0x80U));
        value >>= 7U;
    }
    output.push_back(static_cast<std::byte>(value));
}

inline void appendVarintField(Bytes& output, std::uint32_t field, std::uint64_t value) {
    appendVarint(output, static_cast<std::uint64_t>(field) << 3U);
    appendVarint(output, value);
}

inline void appendBytesField(Bytes& output, std::uint32_t field, std::span<const std::byte> value) {
    appendVarint(output, (static_cast<std::uint64_t>(field) << 3U) | 2U);
    appendVarint(output, value.size());
    output.insert(output.end(), value.begin(), value.end());
}

inline void appendStringField(Bytes& output, std::uint32_t field, std::string_view value) {
    appendBytesField(output, field, std::as_bytes(std::span(value.data(), value.size())));
}

inline Bytes tensorShape(std::span<const std::uint64_t> dimensions) {
    Bytes shape;
    for (const std::uint64_t dimension : dimensions) {
        Bytes dim;
        appendVarintField(dim, 1, dimension);
        appendBytesField(shape, 1, dim);
    }
    return shape;
}

inline Bytes valueInfo(std::string_view name, std::span<const std::uint64_t> dimensions) {
    Bytes tensor_type;
    appendVarintField(tensor_type, 1, 1);  // FLOAT
    const Bytes shape = tensorShape(dimensions);
    appendBytesField(tensor_type, 2, shape);
    Bytes type;
    appendBytesField(type, 1, tensor_type);
    Bytes value;
    appendStringField(value, 1, name);
    appendBytesField(value, 2, type);
    return value;
}

inline Bytes valueInfo(std::string_view name) {
    constexpr std::array dimensions{1ULL, 3ULL, 2ULL, 2ULL};
    return valueInfo(name, dimensions);
}

inline Bytes identityModel(std::string_view node_domain = {}) {
    Bytes node;
    appendStringField(node, 1, "input");
    appendStringField(node, 2, "output");
    appendStringField(node, 4, "Identity");
    if (!node_domain.empty()) appendStringField(node, 7, node_domain);

    Bytes graph;
    appendBytesField(graph, 1, node);
    appendStringField(graph, 2, "cuajone-synthetic");
    const Bytes input = valueInfo("input");
    const Bytes output = valueInfo("output");
    appendBytesField(graph, 11, input);
    appendBytesField(graph, 12, output);

    Bytes opset;
    appendVarintField(opset, 2, 21);
    Bytes model;
    appendVarintField(model, 1, 10);
    appendStringField(model, 2, "cuajone-tests");
    appendBytesField(model, 7, graph);
    appendBytesField(model, 8, opset);
    return model;
}

inline Bytes addModel() {
    Bytes node;
    appendStringField(node, 1, "input");
    appendStringField(node, 1, "bias");
    appendStringField(node, 2, "output");
    appendStringField(node, 4, "Add");

    const std::vector<float> bias(12, 1.0F);
    Bytes tensor;
    for (const std::uint64_t dimension : {1ULL, 3ULL, 2ULL, 2ULL}) {
        appendVarintField(tensor, 1, dimension);
    }
    appendVarintField(tensor, 2, 1);  // FLOAT
    appendStringField(tensor, 8, "bias");
    appendBytesField(tensor, 9, std::as_bytes(std::span(bias)));

    Bytes graph;
    appendBytesField(graph, 1, node);
    appendStringField(graph, 2, "cuajone-synthetic-cuda");
    appendBytesField(graph, 5, tensor);
    const Bytes input = valueInfo("input");
    const Bytes output = valueInfo("output");
    appendBytesField(graph, 11, input);
    appendBytesField(graph, 12, output);

    Bytes opset;
    appendVarintField(opset, 2, 21);
    Bytes model;
    appendVarintField(model, 1, 10);
    appendStringField(model, 2, "cuajone-tests");
    appendBytesField(model, 7, graph);
    appendBytesField(model, 8, opset);
    return model;
}

inline Bytes constantPoseModel() {
    constexpr std::array input_dimensions{1ULL, 3ULL, 640ULL, 640ULL};
    constexpr std::array output_dimensions{1ULL, 1ULL, 56ULL};
    const std::vector<float> values(56, 0.0F);
    Bytes tensor;
    for (const std::uint64_t dimension : output_dimensions) {
        appendVarintField(tensor, 1, dimension);
    }
    appendVarintField(tensor, 2, 1);  // FLOAT
    appendStringField(tensor, 8, "output");
    appendBytesField(tensor, 9, std::as_bytes(std::span(values)));

    Bytes graph;
    appendStringField(graph, 2, "cuajone-synthetic-pose-contract");
    appendBytesField(graph, 5, tensor);
    const Bytes input = valueInfo("input", input_dimensions);
    const Bytes output = valueInfo("output", output_dimensions);
    appendBytesField(graph, 11, input);
    appendBytesField(graph, 12, output);

    Bytes opset;
    appendVarintField(opset, 2, 21);
    Bytes model;
    appendVarintField(model, 1, 10);
    appendStringField(model, 2, "cuajone-tests");
    appendBytesField(model, 7, graph);
    appendBytesField(model, 8, opset);
    return model;
}

inline std::string poseManifest(const std::filesystem::path& model_path, const Bytes& model) {
    return "{\"schema_version\":1,\"artifact_type\":\"onnx\",\"role\":\"pose\","
           "\"model_file\":\"" + model_path.filename().string()
        + "\",\"model_sha256\":\"" + sha256Hex(model) + "\",\"model_size_bytes\":"
        + std::to_string(model.size())
        + ",\"external_data\":false,\"custom_operators\":false,"
          "\"input\":{\"name\":\"input\",\"element_type\":\"float32\",\"shape\":[1,3,640,640]},"
          "\"output\":{\"name\":\"output\",\"element_type\":\"float32\",\"shape\":[1,1,56]},"
          "\"provenance\":{\"source_uri\":\"urn:cuajone:synthetic-pose-contract\","
          "\"exporter\":\"cuajone-tests\",\"license\":\"AGPL-3.0-only\"}}";
}

inline Bytes externalDataModel() {
    Bytes entry;
    appendStringField(entry, 1, "location");
    appendStringField(entry, 2, "weights.bin");
    Bytes tensor;
    appendBytesField(tensor, 13, entry);
    appendVarintField(tensor, 14, 1);

    Bytes model = identityModel();
    Bytes graph;
    appendBytesField(graph, 5, tensor);
    appendBytesField(model, 7, graph);
    return model;
}

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        path_ = std::filesystem::temp_directory_path()
            / ("cuajone_onnx_tests_" + std::to_string(GetCurrentProcessId()));
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
        std::filesystem::create_directory(path_);
    }

    ~TemporaryDirectory() {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }

    const std::filesystem::path& path() const noexcept { return path_; }

private:
    std::filesystem::path path_;
};

inline void writeBytes(const std::filesystem::path& path, std::span<const std::byte> bytes) {
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!output) throw std::runtime_error("Could not write synthetic ONNX test file");
}

inline void writeText(const std::filesystem::path& path, std::string_view text) {
    std::ofstream output(path, std::ios::binary);
    output.write(text.data(), static_cast<std::streamsize>(text.size()));
    if (!output) throw std::runtime_error("Could not write synthetic ONNX manifest");
}

inline std::string manifest(
    const std::filesystem::path& model_path,
    std::span<const std::byte> model,
    std::string_view role = "ppe",
    std::string_view hash = {},
    std::string_view input_name = "input",
    std::string_view extra_root = {}) {
    const std::string digest = hash.empty() ? sha256Hex(model) : std::string(hash);
    return "{\"schema_version\":1,\"artifact_type\":\"onnx\",\"role\":\""
        + std::string(role) + "\",\"model_file\":\"" + model_path.filename().string()
        + "\",\"model_sha256\":\"" + digest + "\",\"model_size_bytes\":"
        + std::to_string(model.size())
        + ",\"external_data\":false,\"custom_operators\":false,"
          "\"input\":{\"name\":\"" + std::string(input_name)
        + "\",\"element_type\":\"float32\",\"shape\":[1,3,2,2]},"
          "\"output\":{\"name\":\"output\",\"element_type\":\"float32\",\"shape\":[1,3,2,2]},"
          "\"provenance\":{\"source_uri\":\"urn:cuajone:synthetic-test\","
          "\"exporter\":\"cuajone-tests\",\"license\":\"AGPL-3.0-only\"}"
        + std::string(extra_root) + "}";
}

inline std::filesystem::path writeModelSet(
    const TemporaryDirectory& directory,
    std::string_view filename,
    const Bytes& model,
    std::string manifest_json = {}) {
    const auto path = directory.path() / filename;
    writeBytes(path, model);
    if (manifest_json.empty()) manifest_json = manifest(path, model);
    writeText(onnxManifestPath(path), manifest_json);
    return path;
}

}  // namespace cuajone::test
