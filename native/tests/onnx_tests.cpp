// SPDX-License-Identifier: AGPL-3.0-only

#include "cuajone/onnx_session.hpp"
#include "cuajone/engine_pipeline.hpp"

#include "onnx_fixture.hpp"

#include <algorithm>
#include <functional>
#include <iostream>
#include <map>
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

void testOutputBindingAliasing() {
    TemporaryDirectory directory;
    const Bytes model = identityModel();
    const auto path = writeModelSet(directory, "aliasing.onnx", model);
    OnnxSession session(path, ModelRole::Ppe);
    const std::vector<float> first_input{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    const std::vector<float> second_input{11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};

    const InferenceOutput first = session.infer(first_input);
    require(std::vector<float>(first.values.begin(), first.values.end()) == first_input,
        "Output-bound inference did not return the input values");
    const float* first_buffer = first.values.data();

    const InferenceOutput second = session.infer(second_input);
    require(std::vector<float>(second.values.begin(), second.values.end()) == second_input,
        "Second output-bound inference did not return the input values");
    require(second.values.data() == first_buffer,
        "Output binding did not reuse the persistent CPU-resident output buffer");

    require(std::vector<float>(first.values.begin(), first.values.end()) == second_input,
        "Stale InferenceOutput did not alias the overwritten session output buffer");
    require(std::ranges::equal(second.shape, std::vector<std::int64_t>({1, 3, 2, 2})),
        "Output binding changed the synthetic output shape");
}

void testSingleIntraOpThreadOption() {
    TemporaryDirectory directory;
    const Bytes model = identityModel();
    const auto path = directory.path() / "pose.onnx";
    writeBytes(path, model);
    writeText(onnxManifestPath(path), manifest(path, model, "pose"));
    OnnxSession session(path, ModelRole::Pose,
        OnnxSessionOptions{OnnxExecutionProvider::Cpu, std::nullopt, 1});
    const std::vector<float> input{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    const InferenceOutput output = session.infer(input);
    require(std::vector<float>(output.values.begin(), output.values.end()) == input,
        "Single-thread ONNX session changed Identity inference values");
    requireThrows([&] {
        OnnxSession invalid(path, ModelRole::Pose,
            OnnxSessionOptions{OnnxExecutionProvider::Cpu, std::nullopt, 0});
    }, "intra-op thread count", "Zero intra-op thread count was accepted");
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

void testPpeManifestBindsSemanticOrder() {
    TemporaryDirectory directory;
    const Bytes model = identityModel();
    const auto path = directory.path() / "wrong-order.onnx";
    writeBytes(path, model);
    std::string json = manifest(path, model);
    const std::string correct = "\"Gloves\",\"Person\"";
    const auto position = json.find(correct);
    require(position != std::string::npos, "Synthetic PPE labels were not found");
    json.replace(position, correct.size(), "\"Person\",\"Gloves\"");
    writeText(onnxManifestPath(path), json);

    EnginePipelineConfig config;
    config.backend = ComputeBackend::Cpu;
    config.provider = InferenceProvider::OnnxRuntimeCpu;
    config.ppe_onnx = path;
    config.ppe_labels = std::map<int, std::string>{{0, "Gloves"}, {1, "Person"},
        {2, "Safety_boots"}, {3, "Vest"}, {4, "respirador"}, {5, "tapaorejas"},
        {6, "Hard_hat"}, {7, "lentes_protectores"}};
    config.analytics.mode = AnalyticsMode::PpeOnly;
    requireThrows([&] { NativeEnginePipeline pipeline(config); },
        "always-all-seven-v2", "Wrong PPE manifest label order was accepted");
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

std::string dynamicManifest(std::string_view role = "ppe", std::size_t schema_version = 2) {
    const std::string ppe = role == "ppe"
        ? R"(,"label_contract":"always-all-seven-v2","labels":["Gloves","Person","Safety_boots","Vest","respirador","tapaorejas","Hard_hat","lentes_protectores"] )"
        : "";
    const std::string output_shape = role == "ppe"
        ? R"([1,12,"predictions"])" : "[1,300,57]";
    const std::string provenance = schema_version == 3
        ? R"({"source_uri":"urn:cuajone:test","exporter":"tests","license":"AGPL-3.0-only","source_checkpoint":{"filename":"model.pt","sha256":")"
            + std::string(64, '0') + R"("}})"
        : R"({"source_uri":"urn:cuajone:test","exporter":"tests","license":"AGPL-3.0-only"})";
    return R"({"schema_version":)" + std::to_string(schema_version) + R"(,"artifact_type":"onnx","role":")"
        + std::string(role)
        + R"(","model_file":"model.onnx","model_sha256":")"
        + std::string(64, '0')
        + R"(","model_size_bytes":1,"external_data":false,"custom_operators":false,)"
          R"("input":{"name":"images","element_type":"float32","shape":[1,3,"height","width"]},)"
          R"("output":{"name":"output0","element_type":"float32","shape":)"
        + output_shape
        + R"(},"provenance":)" + provenance
        + ppe
        + R"(,"dynamic_shape":{"batch":1,"channels":3,"allowed_image_sizes":[640,768,960,1280],"minimum_image_size":640,"optimum_image_size":640,"maximum_image_size":1280}})";
}

void testLegacyPpeManifestRequirements() {
    const std::string labels = R"("label_contract":"always-all-seven-v2","labels":["Gloves","Person","Safety_boots","Vest","respirador","tapaorejas","Hard_hat","lentes_protectores"],)";
    std::string without_labels = manifest(std::filesystem::path("ppe.onnx"), identityModel());
    const auto labels_position = without_labels.find(labels);
    require(labels_position != std::string::npos, "Synthetic PPE label contract was not found");
    without_labels.erase(labels_position, labels.size());
    requireThrows([&] { static_cast<void>(parseOnnxModelManifest(without_labels)); },
        "missing or unsupported fields", "Schema-v1 PPE manifest without labels was accepted");

    std::string unsupported_output = dynamicManifest();
    const std::string approved_output = R"([1,12,"predictions"])";
    const auto output_position = unsupported_output.find(approved_output);
    require(output_position != std::string::npos, "Dynamic PPE output contract was not found");
    unsupported_output.replace(output_position, approved_output.size(), "[1,10,8400]");
    requireThrows([&] { static_cast<void>(parseOnnxModelManifest(unsupported_output)); },
        "approved bounded role schema", "PPE [1,10,8400] contract was accepted with labels");
}

void testDynamicManifestContract() {
    const OnnxModelManifest ppe = parseOnnxModelManifest(dynamicManifest());
    require(ppe.dynamicShapes() && ppe.allowed_image_sizes == std::vector<int>({640, 768, 960, 1280})
            && ppe.input.shape == std::vector<std::int64_t>({1, 3, -1, -1})
            && ppe.output.shape == std::vector<std::int64_t>({1, 12, -1}),
        "Bounded dynamic PPE manifest was not accepted exactly");
    const OnnxModelManifest pose = parseOnnxModelManifest(dynamicManifest("pose"));
    require(pose.dynamicShapes() && pose.output.shape == std::vector<std::int64_t>({1, 300, 57}),
        "Bounded dynamic pose manifest was not accepted");
    const OnnxModelManifest audited_pose = parseOnnxModelManifest(dynamicManifest("pose", 3));
    require(audited_pose.dynamicShapes()
            && audited_pose.source_checkpoint_filename == "model.pt"
            && audited_pose.source_checkpoint_sha256 == std::string(64, '0'),
        "Audited dynamic pose manifest was not accepted");

    std::string unsupported = dynamicManifest();
    const std::string allowed = "[640,768,960,1280]";
    const auto allowed_position = unsupported.find(allowed);
    require(allowed_position != std::string::npos, "Dynamic allowed size fixture was not found");
    unsupported.replace(allowed_position, allowed.size(), "[640,800,960,1280]");
    requireThrows([&] { static_cast<void>(parseOnnxModelManifest(unsupported)); },
        "allowed_image_sizes", "Unsupported dynamic image size was accepted");

    std::string unbounded = dynamicManifest();
    const std::string bounded_output = "[1,12,\"predictions\"]";
    const auto output_position = unbounded.find(bounded_output);
    require(output_position != std::string::npos, "Dynamic output fixture was not found");
    unbounded.replace(output_position, bounded_output.size(), "[1,12,999999]");
    requireThrows([&] { static_cast<void>(parseOnnxModelManifest(unbounded)); },
        "approved bounded role schema", "Unapproved dynamic output formula was accepted");
}

}  // namespace

int main() {
    const std::vector<std::pair<std::string, std::function<void()>>> tests{
        {"verified in-memory inference", testVerifiedInMemoryInference},
        {"output-binding buffer aliasing", testOutputBindingAliasing},
        {"single intra-op thread option", testSingleIntraOpThreadOption},
        {"manifest and artifact boundary", testManifestAndArtifactBoundary},
        {"PPE manifest semantic order", testPpeManifestBindsSemanticOrder},
        {"legacy PPE manifest requirements", testLegacyPpeManifestRequirements},
        {"external data and custom operators", testExternalDataAndCustomOperatorsRejected},
        {"manifest resource limits", testManifestResourceLimits},
        {"bounded dynamic manifest contract", testDynamicManifestContract},
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
