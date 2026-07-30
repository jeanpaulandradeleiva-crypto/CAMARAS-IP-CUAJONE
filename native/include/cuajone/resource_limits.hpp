// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

namespace cuajone::resource_limits {

inline constexpr std::size_t kMaximumManifestBytes = 64U * 1024U;
inline constexpr std::size_t kMaximumOnnxModelBytes = 256U * 1024U * 1024U;
inline constexpr std::size_t kMaximumTensorRtEngineBytes = 1024ULL * 1024ULL * 1024ULL;
inline constexpr std::size_t kMaximumTensorRank = 8;
inline constexpr std::int64_t kMaximumTensorDimension = 1'000'000;
inline constexpr int kMaximumImageDimension = 4096;
inline constexpr std::size_t kMaximumInputElements =
    3ULL * kMaximumImageDimension * kMaximumImageDimension;
inline constexpr std::size_t kMaximumOutputElements = 16U * 1024U * 1024U;
inline constexpr std::size_t kMaximumTensorBytes = 256U * 1024U * 1024U;

inline std::size_t checkedMultiply(
    std::size_t left,
    std::size_t right,
    std::size_t maximum,
    std::string_view name) {
    if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
        throw std::overflow_error(std::string(name) + " size overflow");
    }
    const std::size_t result = left * right;
    if (result > maximum) {
        throw std::runtime_error(std::string(name) + " exceeds the supported resource limit");
    }
    return result;
}

inline std::size_t checkedVolume(
    std::span<const std::int64_t> shape,
    std::size_t maximum_elements,
    std::string_view name) {
    if (shape.empty() || shape.size() > kMaximumTensorRank) {
        throw std::runtime_error(std::string(name) + " rank is outside the supported range");
    }
    std::size_t result = 1;
    for (const std::int64_t dimension : shape) {
        if (dimension <= 0 || dimension > kMaximumTensorDimension) {
            throw std::runtime_error(
                std::string(name) + " contains a dynamic or out-of-range dimension");
        }
        result = checkedMultiply(
            result, static_cast<std::size_t>(dimension), maximum_elements, name);
    }
    return result;
}

inline std::size_t checkedTensorBytes(
    std::size_t elements,
    std::size_t element_size,
    std::string_view name) {
    return checkedMultiply(elements, element_size, kMaximumTensorBytes, name);
}

}  // namespace cuajone::resource_limits
