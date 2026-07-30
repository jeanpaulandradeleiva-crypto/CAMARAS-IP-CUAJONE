// SPDX-License-Identifier: AGPL-3.0-only

#include "cuajone/model_manifest.hpp"
#include "cuajone/onnx_session.hpp"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Bytes = std::vector<std::byte>;
using namespace cuajone;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename Function>
void requireThrows(Function function, const std::string& expected, const std::string& message) {
    try {
        function();
    } catch (const std::exception& error) {
        if (std::string(error.what()).find(expected) != std::string::npos) return;
        throw std::runtime_error(message + "; unexpected error: " + error.what());
    }
    throw std::runtime_error(message + "; no exception was thrown");
}

void appendVarint(Bytes& output, std::uint64_t value) {
    while (value >= 0x80U) {
        output.push_back(static_cast<std::byte>((value & 0x7FU) | 0x80U));
        value >>= 7U;
    }
    output.push_back(static_cast<std::byte>(value));
}

void appendVarintField(Bytes& output, std::uint32_t field, std::uint64_t value) {
    appendVarint(output, static_cast<std::uint64_t>(field) << 3U);
    appendVarint(output, value);
}

void appendBytesField(Bytes& output, std::uint32_t field, std::span<const std::byte> value) {
    appendVarint(output, (static_cast<std::uint64_t>(field) << 3U) | 2U);
    appendVarint(output, value.size());
    output.insert(output.end(), value.begin(), value.end());
}

void appendStringField(Bytes& output, std::uint32_t field, std::string_view value) {
    appendBytesField(output, field, std::as_bytes(std::span(value.data(), value.size())));
}

Bytes tensorShape() {
    Bytes shape;
    for (const std::uint64_t dimension : {1ULL, 3ULL, 2ULL, 2ULL}) {
        Bytes dim;
        appendVarintField(dim, 1, dimension);
        appendBytesField(shape, 1, dim);
    }
    return shape;
}

Bytes valueInfo(std::string_view name) {
    Bytes tensor_type;
    appendVarintField(tensor_type, 1, 1);  // FLOAT
    const Bytes shape = tensorShape();
    appendBytesField(tensor_type, 2, shape);
    Bytes type;
    appendBytesField(type, 1, tensor_type);
    Bytes value;
    appendStringField(value, 1, name);
    appendBytesField(value, 2, type);
    return value;
}

Bytes identityModel(std::string_view node_domain = {}) {
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

Bytes externalDataModel() {
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

void writeBytes(const std::filesystem::path& path, std::span<const std::byte> bytes) {
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!output) throw std::runtime_error("Could not write synthetic ONNX test file");
}

void writeText(const std::filesystem::path& path, std::string_view text) {
    std::ofstream output(path, std::ios::binary);
    output.write(text.data(), static_cast<std::streamsize>(text.size()));
    if (!output) throw std::runtime_error("Could not write synthetic ONNX manifest");
}

std::string manifest(
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

std::filesystem::path writeModelSet(
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

void testVerifiedInMemoryInference() {
    TemporaryDirectory directory;
    const Bytes model = identityModel();
    const auto path = writeModelSet(directory, "ppe.onnx", model);
    OnnxCpuSession session(path, ModelRole::Ppe);
    const std::vector<float> input{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    const InferenceOutput output = session.infer(input);
    require(std::ranges::equal(output.shape, std::vector<std::int64_t>({1, 3, 2, 2})),
        "Synthetic output shape changed");
    require(std::vector<float>(output.values.begin(), output.values.end()) == input,
        "Synthetic in-memory ONNX inference changed Identity values");
}

void testManifestAndArtifactBoundary() {
    TemporaryDirectory directory;
    const Bytes model = identityModel();
    const auto missing = directory.path() / "missing.onnx";
    writeBytes(missing, model);
    requireThrows([&] { OnnxCpuSession session(missing, ModelRole::Ppe); },
        "manifest must be", "ONNX without a manifest was accepted");

    const auto renamed = directory.path() / "renamed.engine";
    writeBytes(renamed, model);
    requireThrows([&] { OnnxCpuSession session(renamed, ModelRole::Ppe); },
        ".onnx extension", "ONNX bytes renamed as .engine were accepted by CPU");

    const auto role_path = directory.path() / "role.onnx";
    writeBytes(role_path, model);
    writeText(onnxManifestPath(role_path), manifest(role_path, model, "pose"));
    requireThrows([&] { OnnxCpuSession session(role_path, ModelRole::Ppe); },
        "expected ppe", "Pose manifest was accepted as PPE");

    const auto hash_path = directory.path() / "hash.onnx";
    writeBytes(hash_path, model);
    writeText(onnxManifestPath(hash_path), manifest(hash_path, model, "ppe", std::string(64, '0')));
    requireThrows([&] { OnnxCpuSession session(hash_path, ModelRole::Ppe); },
        "SHA-256", "ONNX hash mismatch was accepted");

    const auto size_path = directory.path() / "size.onnx";
    writeBytes(size_path, model);
    std::string wrong_size = manifest(size_path, model);
    const std::string size_field = "\"model_size_bytes\":" + std::to_string(model.size());
    const auto size_position = wrong_size.find(size_field);
    require(size_position != std::string::npos, "Synthetic manifest size field was not found");
    wrong_size.replace(size_position, size_field.size(),
        "\"model_size_bytes\":" + std::to_string(model.size() + 1));
    writeText(onnxManifestPath(size_path), wrong_size);
    requireThrows([&] { OnnxCpuSession session(size_path, ModelRole::Ppe); },
        "byte size", "ONNX size mismatch was accepted");

    const auto unknown_path = directory.path() / "unknown.onnx";
    writeBytes(unknown_path, model);
    writeText(onnxManifestPath(unknown_path), manifest(unknown_path, model, "ppe", {}, "input", ",\"unknown\":1"));
    requireThrows([&] { OnnxCpuSession session(unknown_path, ModelRole::Ppe); },
        "unsupported fields", "Unknown manifest field was accepted");

    const auto io_path = directory.path() / "io.onnx";
    writeBytes(io_path, model);
    writeText(onnxManifestPath(io_path), manifest(io_path, model, "ppe", {}, "wrong_input"));
    requireThrows([&] { OnnxCpuSession session(io_path, ModelRole::Ppe); },
        "does not exactly match", "Manifest I/O mismatch was accepted");
}

void testExternalDataAndCustomOperatorsRejected() {
    TemporaryDirectory directory;
    const Bytes external = externalDataModel();
    const auto external_path = writeModelSet(directory, "external.onnx", external);
    writeText(directory.path() / "weights.bin", "untrusted-sidecar");
    requireThrows([&] { OnnxCpuSession session(external_path, ModelRole::Ppe); },
        "external_data", "External-data ONNX was accepted");

    const Bytes custom = identityModel("com.example.custom");
    const auto custom_path = writeModelSet(directory, "custom.onnx", custom);
    requireThrows([&] { OnnxCpuSession session(custom_path, ModelRole::Ppe); },
        "custom operator domain", "Custom operator domain was accepted");
}

void testManifestResourceLimits() {
    TemporaryDirectory directory;
    const Bytes model = identityModel();
    const auto path = directory.path() / "oversized.onnx";
    writeBytes(path, model);
    std::string json = manifest(path, model);
    const std::string supported = "[1,3,2,2]";
    const auto position = json.find(supported);
    require(position != std::string::npos, "Synthetic manifest input shape was not found");
    json.replace(position, supported.size(), "[1,3,4097,1]");
    writeText(onnxManifestPath(path), json);
    requireThrows([&] { OnnxCpuSession session(path, ModelRole::Ppe); },
        "bounded batch-1", "Oversized ONNX input contract was accepted");

    const auto output_path = directory.path() / "oversized-output.onnx";
    writeBytes(output_path, model);
    json = manifest(output_path, model);
    const auto first_shape = json.find(supported);
    const auto output_shape = json.find(supported, first_shape + supported.size());
    require(output_shape != std::string::npos, "Synthetic manifest output shape was not found");
    json.replace(output_shape, supported.size(), "[1,4096,4097]");
    writeText(onnxManifestPath(output_path), json);
    requireThrows([&] { OnnxCpuSession session(output_path, ModelRole::Ppe); },
        "resource limit", "Oversized ONNX output contract was accepted");
}

}  // namespace

int main() {
    const std::vector<std::pair<std::string, std::function<void()>>> tests{
        {"verified in-memory inference", testVerifiedInMemoryInference},
        {"manifest and artifact boundary", testManifestAndArtifactBoundary},
        {"external data and custom operators", testExternalDataAndCustomOperatorsRejected},
        {"manifest resource limits", testManifestResourceLimits},
    };
    int failures = 0;
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
              << tests.size() << " ONNX tests passed\n";
    return failures == 0 ? 0 : 1;
}
