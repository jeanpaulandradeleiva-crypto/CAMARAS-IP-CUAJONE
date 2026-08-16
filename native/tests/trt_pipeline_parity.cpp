// SPDX-License-Identifier: AGPL-3.0-only

#include "cuajone/contracts.hpp"
#include "cuajone/engine_pipeline.hpp"
#include "cuajone/tensorrt_runtime.hpp"

#include <BYTETracker.h>

#include <opencv2/imgcodecs.hpp>

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: cuajone_trt_pipeline_parity <ppe.engine> <pose.engine> <image>\n";
        return 2;
    }
    try {
        cuajone::selectCudaDevice(std::nullopt);
        const std::filesystem::path ppe_engine{argv[1]};
        const std::filesystem::path pose_engine{argv[2]};
        const cv::Mat frame = cv::imread(argv[3], cv::IMREAD_COLOR);
        if (frame.empty() || frame.type() != CV_8UC3) {
            throw std::runtime_error("Could not decode the parity image as CV_8UC3");
        }

        const auto make_config = [&](bool serial) {
            cuajone::EnginePipelineConfig config;
            config.backend = cuajone::ComputeBackend::Cuda;
            config.provider = cuajone::InferenceProvider::TensorRt;
            config.ppe_engine = ppe_engine;
            config.pose_engine = pose_engine;
            config.analytics.mode = cuajone::AnalyticsMode::PpeFall;
#ifdef CUAJONE_INTERNAL_DIAGNOSTICS
            config.force_serial_tensorrt = serial;
#else
            static_cast<void>(serial);
#endif
            return config;
        };

        constexpr std::uint64_t frame_count = 5;
        std::vector<std::string> serial_canonical;
        serial_canonical.reserve(frame_count);
        {
            cuajone::NativeEnginePipeline pipeline(make_config(true));
            for (std::uint64_t frame_id = 1; frame_id <= frame_count; ++frame_id) {
                const auto processed = pipeline.processFrame(
                    frame, "trt-parity", frame_id,
                    1000 + static_cast<std::int64_t>(frame_id), "2026-08-01T00:00:00Z");
                serial_canonical.push_back(cuajone::canonicalJson(processed.canonical));
            }
        }
        BaseTrack::reset_count();
        {
            cuajone::NativeEnginePipeline pipeline(make_config(false));
            for (std::uint64_t frame_id = 1; frame_id <= frame_count; ++frame_id) {
                const auto processed = pipeline.processFrame(
                    frame, "trt-parity", frame_id,
                    1000 + static_cast<std::int64_t>(frame_id), "2026-08-01T00:00:00Z");
                const std::string canonical = cuajone::canonicalJson(processed.canonical);
                if (canonical != serial_canonical.at(static_cast<std::size_t>(frame_id - 1))) {
                    std::cerr << "Parity mismatch at frame " << frame_id
                              << " between sequential and overlap TensorRT pipelines\n";
                    return 1;
                }
            }
        }
        std::cout << "TensorRT pipeline parity passed: sequential == overlap over "
                  << frame_count << " frames\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
