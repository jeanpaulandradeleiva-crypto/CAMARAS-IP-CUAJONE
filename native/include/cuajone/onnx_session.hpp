// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include "cuajone/inference_session.hpp"
#include "cuajone/model_manifest.hpp"

#include <filesystem>
#include <memory>

namespace cuajone {

class OnnxCpuSession final : public InferenceSession {
public:
    OnnxCpuSession(const std::filesystem::path& model_path, ModelRole expected_role);
    ~OnnxCpuSession() override;
    OnnxCpuSession(const OnnxCpuSession&) = delete;
    OnnxCpuSession& operator=(const OnnxCpuSession&) = delete;

    [[nodiscard]] int inputWidth() const noexcept override;
    [[nodiscard]] int inputHeight() const noexcept override;
    [[nodiscard]] const std::vector<std::int64_t>& outputShape() const noexcept override;
    InferenceOutput infer(std::span<const float> nchw_input) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace cuajone
