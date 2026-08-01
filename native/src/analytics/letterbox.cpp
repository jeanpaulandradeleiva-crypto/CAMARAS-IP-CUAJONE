// SPDX-License-Identifier: AGPL-3.0-only

#include "cuajone/letterbox.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace cuajone {
namespace {

float clampCoordinate(float value, int maximum) noexcept {
    return std::clamp(value, 0.0F, static_cast<float>(maximum));
}

}  // namespace

Point LetterboxTransform::restore(Point point) const noexcept {
    return {
        clampCoordinate((point.x - static_cast<float>(padding_left)) / scale_x, source_width),
        clampCoordinate((point.y - static_cast<float>(padding_top)) / scale_y, source_height),
    };
}

Box LetterboxTransform::restore(Box box) const noexcept {
    const Point top_left = restore(Point{box.x1, box.y1});
    const Point bottom_right = restore(Point{box.x2, box.y2});
    return {top_left.x, top_left.y, bottom_right.x, bottom_right.y};
}

LetterboxTransform makeLetterboxTransform(
    int source_width,
    int source_height,
    int model_width,
    int model_height) {
    if (source_width <= 0 || source_height <= 0 || model_width <= 0 || model_height <= 0) {
        throw std::invalid_argument("Source and model dimensions must be positive");
    }
    const float ratio = std::min(
        static_cast<float>(model_width) / static_cast<float>(source_width),
        static_cast<float>(model_height) / static_cast<float>(source_height));
    const int resized_width = std::max(1, static_cast<int>(std::round(source_width * ratio)));
    const int resized_height = std::max(1, static_cast<int>(std::round(source_height * ratio)));
    const int left = static_cast<int>(std::round(
        static_cast<float>(model_width - resized_width) / 2.0F - 0.1F));
    const int top = static_cast<int>(std::round(
        static_cast<float>(model_height - resized_height) / 2.0F - 0.1F));
    return {
        source_width,
        source_height,
        model_width,
        model_height,
        static_cast<float>(resized_width) / static_cast<float>(source_width),
        static_cast<float>(resized_height) / static_cast<float>(source_height),
        left,
        top,
    };
}

}  // namespace cuajone
