// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include "cuajone/engine_reader.hpp"
#include "cuajone/yolo_decode.hpp"

#include <NvInfer.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace cuajone {

void checkCuda(cudaError_t result, std::string_view operation);

class CudaStream {
public:
    CudaStream();
    ~CudaStream();
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    CudaStream(CudaStream&& other) noexcept;
    CudaStream& operator=(CudaStream&& other) noexcept;
    [[nodiscard]] cudaStream_t get() const noexcept;

private:
    cudaStream_t stream_{};
};

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t bytes);
    ~DeviceBuffer();
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept;
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;
    [[nodiscard]] void* data() noexcept;
    [[nodiscard]] std::size_t size() const noexcept;

private:
    void* data_{};
    std::size_t size_{};
};

class TensorRtLogger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* message) noexcept override;
};

struct InferenceOutput {
    std::span<const float> values;
    std::span<const std::int64_t> shape;
};

class TensorRtSession {
public:
    TensorRtSession(const EngineFile& engine_file, std::optional<std::array<int, 2>> preferred_image_size);

    TensorRtSession(const TensorRtSession&) = delete;
    TensorRtSession& operator=(const TensorRtSession&) = delete;

    [[nodiscard]] int inputWidth() const noexcept;
    [[nodiscard]] int inputHeight() const noexcept;
    [[nodiscard]] const std::vector<std::int64_t>& outputShape() const noexcept;
    InferenceOutput infer(std::span<const float> nchw_input);

private:
    static std::size_t elementSize(nvinfer1::DataType type);
    static std::size_t volume(const nvinfer1::Dims& dimensions);

    TensorRtLogger logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    CudaStream stream_;
    std::string input_name_;
    std::string output_name_;
    nvinfer1::DataType input_type_{};
    nvinfer1::DataType output_type_{};
    int input_width_{};
    int input_height_{};
    std::size_t input_elements_{};
    std::size_t output_elements_{};
    DeviceBuffer input_buffer_;
    DeviceBuffer output_buffer_;
    std::vector<float> host_input_float_;
    std::vector<__half> host_input_half_;
    std::vector<float> host_output_float_;
    std::vector<__half> host_output_half_;
    std::vector<float> float_output_;
    std::vector<std::int64_t> output_shape_;
};

struct DeviceSummary {
    int count{};
    int selected{};
    std::string name;
    std::size_t total_memory{};
    int compute_major{};
    int compute_minor{};
};

DeviceSummary selectCudaDevice(int device);

}  // namespace cuajone
