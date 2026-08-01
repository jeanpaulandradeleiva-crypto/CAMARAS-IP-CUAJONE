// SPDX-License-Identifier: AGPL-3.0-only

#include "cuajone/onnx_session.hpp"

#include "onnx_fixture.hpp"

#include <algorithm>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using namespace cuajone;
using namespace cuajone::test;

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

void testVerifiedInMemoryInference() {
    TemporaryDirectory directory;
    const Bytes model = identityModel();
    const auto path = writeModelSet(directory, "ppe.onnx", model);
    OnnxSession session(path, ModelRole::Ppe);
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
    requireThrows([&] { OnnxSession session(missing, ModelRole::Ppe); },
        "manifest must be", "ONNX without a manifest was accepted");

    const auto renamed = directory.path() / "renamed.engine";
    writeBytes(renamed, model);
    requireThrows([&] { OnnxSession session(renamed, ModelRole::Ppe); },
        ".onnx extension", "ONNX bytes renamed as .engine were accepted by CPU");

    const auto role_path = directory.path() / "role.onnx";
    writeBytes(role_path, model);
    writeText(onnxManifestPath(role_path), manifest(role_path, model, "pose"));
    requireThrows([&] { OnnxSession session(role_path, ModelRole::Ppe); },
        "expected ppe", "Pose manifest was accepted as PPE");

    const auto hash_path = directory.path() / "hash.onnx";
    writeBytes(hash_path, model);
    writeText(onnxManifestPath(hash_path), manifest(hash_path, model, "ppe", std::string(64, '0')));
    requireThrows([&] { OnnxSession session(hash_path, ModelRole::Ppe); },
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
    requireThrows([&] { OnnxSession session(size_path, ModelRole::Ppe); },
        "byte size", "ONNX size mismatch was accepted");

    const auto unknown_path = directory.path() / "unknown.onnx";
    writeBytes(unknown_path, model);
    writeText(onnxManifestPath(unknown_path), manifest(unknown_path, model, "ppe", {}, "input", ",\"unknown\":1"));
    requireThrows([&] { OnnxSession session(unknown_path, ModelRole::Ppe); },
        "unsupported fields", "Unknown manifest field was accepted");

    const auto io_path = directory.path() / "io.onnx";
    writeBytes(io_path, model);
    writeText(onnxManifestPath(io_path), manifest(io_path, model, "ppe", {}, "wrong_input"));
    requireThrows([&] { OnnxSession session(io_path, ModelRole::Ppe); },
        "does not exactly match", "Manifest I/O mismatch was accepted");
}

void testExternalDataAndCustomOperatorsRejected() {
    TemporaryDirectory directory;
    const Bytes external = externalDataModel();
    const auto external_path = writeModelSet(directory, "external.onnx", external);
    writeText(directory.path() / "weights.bin", "untrusted-sidecar");
    requireThrows([&] { OnnxSession session(external_path, ModelRole::Ppe); },
        "external_data", "External-data ONNX was accepted");

    const Bytes custom = identityModel("com.example.custom");
    const auto custom_path = writeModelSet(directory, "custom.onnx", custom);
    requireThrows([&] { OnnxSession session(custom_path, ModelRole::Ppe); },
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
    requireThrows([&] { OnnxSession session(path, ModelRole::Ppe); },
        "bounded batch-1", "Oversized ONNX input contract was accepted");

    const auto output_path = directory.path() / "oversized-output.onnx";
    writeBytes(output_path, model);
    json = manifest(output_path, model);
    const auto first_shape = json.find(supported);
    const auto output_shape = json.find(supported, first_shape + supported.size());
    require(output_shape != std::string::npos, "Synthetic manifest output shape was not found");
    json.replace(output_shape, supported.size(), "[1,4096,4097]");
    writeText(onnxManifestPath(output_path), json);
    requireThrows([&] { OnnxSession session(output_path, ModelRole::Ppe); },
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
