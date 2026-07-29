// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include "cuajone/types.hpp"

namespace cuajone {

struct LetterboxTransform {
    int source_width{};
    int source_height{};
    int model_width{};
    int model_height{};
    float scale_x{};
    float scale_y{};
    int padding_left{};
    int padding_top{};

    [[nodiscard]] Point restore(Point point) const noexcept;
    [[nodiscard]] Box restore(Box box) const noexcept;
};

}  // namespace cuajone
