// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace cuajone {

struct InferenceOutput {
    std::span<const float> values;
    std::span<const std::int64_t> shape;
};

class InferenceSession {
public:
    virtual ~InferenceSession() = default;
    [[nodiscard]] virtual int inputWidth() const noexcept = 0;
    [[nodiscard]] virtual int inputHeight() const noexcept = 0;
    [[nodiscard]] virtual const std::vector<std::int64_t>& outputShape() const noexcept = 0;
    virtual InferenceOutput infer(std::span<const float> nchw_input) = 0;
};

}  // namespace cuajone
