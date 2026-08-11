// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace cuajone {

enum class ModelRole {
    Ppe,
    Pose,
};

struct TensorContract {
    std::string name;
    std::vector<std::int64_t> shape;
};

struct OnnxModelManifest {
    std::size_t schema_version{1};
    ModelRole role{ModelRole::Ppe};
    std::string model_file;
    std::string model_sha256;
    std::size_t model_size_bytes{};
    std::string source_uri;
    std::string exporter;
    std::string license;
    std::string label_contract;
    std::vector<std::string> labels;
    TensorContract input;
    TensorContract output;
    std::vector<int> allowed_image_sizes;
    int minimum_image_size{};
    int optimum_image_size{};
    int maximum_image_size{};

    [[nodiscard]] bool dynamicShapes() const noexcept { return schema_version == 2; }
};

struct VerifiedOnnxModel {
    OnnxModelManifest manifest;
    std::filesystem::path manifest_path;
    std::vector<std::byte> bytes;
};

[[nodiscard]] std::filesystem::path onnxManifestPath(const std::filesystem::path& model_path);
[[nodiscard]] std::string_view modelRoleName(ModelRole role) noexcept;
[[nodiscard]] OnnxModelManifest parseOnnxModelManifest(std::string_view json);
[[nodiscard]] std::string sha256Hex(std::span<const std::byte> bytes);
[[nodiscard]] VerifiedOnnxModel verifyOnnxModel(
    const std::filesystem::path& model_path,
    ModelRole expected_role);
void validateOnnxModelSecurity(std::span<const std::byte> model_bytes);

}  // namespace cuajone
