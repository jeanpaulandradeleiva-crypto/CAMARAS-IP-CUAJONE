// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string_view>

namespace cuajone {

inline constexpr std::array<int, 4> kAllowedImageSizes{640, 768, 960, 1280};
inline constexpr int kDefaultImageSize = 640;
inline constexpr int kMaximumImageSize = 1280;
inline constexpr std::array<std::string_view, 8> kPpeOutputLabels{
    "Gloves", "Person", "Safety_boots", "Vest", "respirador", "tapaorejas",
    "Hard_hat", "lentes_protectores",
};
inline constexpr float kDefaultPpeConfidence = 0.30F;

[[nodiscard]] inline bool isSupportedImageSize(int value) noexcept {
    return std::ranges::find(kAllowedImageSizes, value) != kAllowedImageSizes.end();
}

inline void validateImageSize(int value) {
    if (!isSupportedImageSize(value)) {
        throw std::invalid_argument("imgsz must be one of 640, 768, 960, or 1280");
    }
}

inline void validatePpeClassConfidences(const std::array<float, 8>& values) {
    if (std::ranges::any_of(values, [](float value) {
            return !std::isfinite(value) || value < 0.0F || value > 1.0F;
        })) {
        throw std::invalid_argument("PPE class confidence thresholds must be finite and in [0, 1]");
    }
}

[[nodiscard]] inline std::size_t yoloPredictionCount(int image_size) {
    validateImageSize(image_size);
    const auto cells = [image_size](int stride) {
        const std::size_t side = static_cast<std::size_t>(image_size / stride);
        return side * side;
    };
    return cells(8) + cells(16) + cells(32);
}

}  // namespace cuajone
