// SPDX-License-Identifier: AGPL-3.0-only

#include "cuajone/letterbox.hpp"

#include <algorithm>

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

}  // namespace cuajone
