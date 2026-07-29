// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace cuajone {

struct EngineMetadata {
    std::optional<std::string> task;
    std::map<int, std::string> names;
    std::optional<std::array<int, 2>> image_size;
    std::optional<std::array<int, 2>> keypoint_shape;
};

class EngineFile {
public:
    static EngineFile read(const std::filesystem::path& path);

    [[nodiscard]] std::span<const std::byte> plan() const noexcept;
    [[nodiscard]] const EngineMetadata& metadata() const noexcept;
    [[nodiscard]] bool hasMetadataPrefix() const noexcept;

private:
    std::vector<std::byte> bytes_;
    std::size_t plan_offset_{};
    EngineMetadata metadata_;
    bool has_metadata_prefix_{};
};

}  // namespace cuajone
