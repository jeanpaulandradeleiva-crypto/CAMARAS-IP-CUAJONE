// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include "cuajone/inference_session.hpp"
#include "cuajone/inference_settings.hpp"
#include "cuajone/model_manifest.hpp"

#include <filesystem>
#include <memory>
#include <optional>

namespace cuajone {

enum class OnnxExecutionProvider {
    Cpu,
    Cuda,
};

struct OnnxSessionOptions {
    OnnxExecutionProvider execution_provider{OnnxExecutionProvider::Cpu};
    std::optional<int> cuda_device;
    std::optional<int> intra_op_num_threads;
};

class OnnxSession final : public InferenceSession {
public:
    OnnxSession(
        const std::filesystem::path& model_path,
        ModelRole expected_role,
        OnnxSessionOptions options = {},
        std::optional<int> image_size = std::nullopt);
    ~OnnxSession() override;
    OnnxSession(const OnnxSession&) = delete;
    OnnxSession& operator=(const OnnxSession&) = delete;

    [[nodiscard]] int inputWidth() const noexcept override;
    [[nodiscard]] int inputHeight() const noexcept override;
    [[nodiscard]] const std::vector<std::int64_t>& outputShape() const noexcept override;
    [[nodiscard]] const OnnxModelManifest& manifest() const noexcept;
    InferenceOutput infer(std::span<const float> nchw_input) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace cuajone
