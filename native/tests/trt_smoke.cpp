// SPDX-License-Identifier: AGPL-3.0-only

#include "cuajone/engine_reader.hpp"
#include "cuajone/tensorrt_runtime.hpp"

#include <filesystem>
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: cuajone_trt_smoke <engine>\n";
        return 2;
    }
    try {
        cuajone::selectCudaDevice(0);
        const auto file = cuajone::EngineFile::read(std::filesystem::path(argv[1]));
        const cuajone::TensorRtSession session(file, file.metadata().image_size);
        std::cout << "TensorRT engine inspected: input " << session.inputWidth() << 'x'
                  << session.inputHeight() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
