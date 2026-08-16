// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include "cuajone/engine_reader.hpp"
#include "cuajone/inference_session.hpp"
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

class PinnedHostBuffer {
public:
    explicit PinnedHostBuffer(std::size_t bytes);
    ~PinnedHostBuffer();
    PinnedHostBuffer(const PinnedHostBuffer&) = delete;
    PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;
    PinnedHostBuffer(PinnedHostBuffer&& other) noexcept;
    PinnedHostBuffer& operator=(PinnedHostBuffer&& other) noexcept;
    [[nodiscard]] void* data() noexcept;
    [[nodiscard]] const void* data() const noexcept;
    [[nodiscard]] std::size_t size() const noexcept;

private:
    void* data_{};
    std::size_t size_{};
};

class TensorRtLogger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* message) noexcept override;
};

class TensorRtSession final : public InferenceSession {
public:
    TensorRtSession(const EngineFile& engine_file, std::optional<std::array<int, 2>> preferred_image_size);
    ~TensorRtSession() override;

    TensorRtSession(const TensorRtSession&) = delete;
    TensorRtSession& operator=(const TensorRtSession&) = delete;

    [[nodiscard]] int inputWidth() const noexcept override;
    [[nodiscard]] int inputHeight() const noexcept override;
    [[nodiscard]] const std::vector<std::int64_t>& outputShape() const noexcept override;
    InferenceOutput infer(std::span<const float> nchw_input) override;

    void submit(std::span<const float> nchw_input);
    InferenceOutput collect();

private:
    static std::size_t elementSize(nvinfer1::DataType type);
    static std::size_t volume(
        const nvinfer1::Dims& dimensions,
        std::size_t maximum_elements,
        std::string_view name);

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
    PinnedHostBuffer host_input_;
    PinnedHostBuffer host_output_;
    std::vector<float> float_output_;
    cudaEvent_t event_{};
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

DeviceSummary selectCudaDevice(std::optional<int> device);

}  // namespace cuajone
