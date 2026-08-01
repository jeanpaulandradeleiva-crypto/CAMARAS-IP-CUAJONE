// SPDX-License-Identifier: AGPL-3.0-only

#include "cuajone/types.hpp"

#include <algorithm>

namespace cuajone {

float boxArea(const Box& box) noexcept {
    return std::max(0.0F, box.x2 - box.x1) * std::max(0.0F, box.y2 - box.y1);
}

float intersectionOverUnion(const Box& lhs, const Box& rhs) noexcept {
    const float x1 = std::max(lhs.x1, rhs.x1);
    const float y1 = std::max(lhs.y1, rhs.y1);
    const float x2 = std::min(lhs.x2, rhs.x2);
    const float y2 = std::min(lhs.y2, rhs.y2);
    const float intersection = std::max(0.0F, x2 - x1) * std::max(0.0F, y2 - y1);
    const float union_area = boxArea(lhs) + boxArea(rhs) - intersection;
    return union_area > 0.0F ? intersection / union_area : 0.0F;
}

}  // namespace cuajone
