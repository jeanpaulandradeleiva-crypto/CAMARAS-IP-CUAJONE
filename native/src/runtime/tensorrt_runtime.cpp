// SPDX-License-Identifier: AGPL-3.0-only

#include "cuajone/tensorrt_runtime.hpp"
#include "cuajone/compute.hpp"
#include "cuajone/resource_limits.hpp"

#include <algorithm>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>

static_assert(NV_TENSORRT_MAJOR == 11, "TensorRT major version 11 is required");

namespace cuajone {

void checkCuda(cudaError_t result, std::string_view operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
    }
}

CudaStream::CudaStream() {
    checkCuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
}

CudaStream::~CudaStream() {
    if (stream_ != nullptr) cudaStreamDestroy(stream_);
}

CudaStream::CudaStream(CudaStream&& other) noexcept
    : stream_(std::exchange(other.stream_, nullptr)) {}

CudaStream& CudaStream::operator=(CudaStream&& other) noexcept {
    if (this == &other) return *this;
    if (stream_ != nullptr) cudaStreamDestroy(stream_);
    stream_ = std::exchange(other.stream_, nullptr);
    return *this;
}

cudaStream_t CudaStream::get() const noexcept {
    return stream_;
}

DeviceBuffer::DeviceBuffer(std::size_t bytes) : size_(bytes) {
    if (bytes > 0) checkCuda(cudaMalloc(&data_, bytes), "cudaMalloc");
}

DeviceBuffer::~DeviceBuffer() {
    if (data_ != nullptr) cudaFree(data_);
}

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0)) {}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
    if (this == &other) return *this;
    if (data_ != nullptr) cudaFree(data_);
    data_ = std::exchange(other.data_, nullptr);
    size_ = std::exchange(other.size_, 0);
    return *this;
}

void* DeviceBuffer::data() noexcept { return data_; }
std::size_t DeviceBuffer::size() const noexcept { return size_; }

void TensorRtLogger::log(Severity severity, const char* message) noexcept {
    if (severity <= Severity::kWARNING) {
        std::cerr << "TensorRT: " << message << '\n';
    }
}

std::size_t TensorRtSession::elementSize(nvinfer1::DataType type) {
    switch (type) {
        case nvinfer1::DataType::kFLOAT: return sizeof(float);
        case nvinfer1::DataType::kHALF: return sizeof(__half);
        default: throw std::runtime_error("Only FP32 and FP16 TensorRT I/O tensors are supported");
    }
}

std::size_t TensorRtSession::volume(
    const nvinfer1::Dims& dimensions,
    std::size_t maximum_elements,
    std::string_view name) {
    if (dimensions.nbDims <= 0
        || static_cast<std::size_t>(dimensions.nbDims) > resource_limits::kMaximumTensorRank) {
        throw std::runtime_error(std::string(name) + " rank is outside the supported range");
    }
    std::size_t result = 1;
    for (int index = 0; index < dimensions.nbDims; ++index) {
        if (dimensions.d[index] <= 0
            || dimensions.d[index] > resource_limits::kMaximumTensorDimension) {
            throw std::runtime_error(std::string(name) + " has an unresolved or out-of-range dimension");
        }
        result = resource_limits::checkedMultiply(
            result, static_cast<std::size_t>(dimensions.d[index]), maximum_elements, name);
    }
    return result;
}

TensorRtSession::TensorRtSession(
    const EngineFile& engine_file,
    std::optional<std::array<int, 2>> preferred_image_size)
    : runtime_(nvinfer1::createInferRuntime(logger_)),
      stream_(),
      input_buffer_(0),
      output_buffer_(0) {
    if (!runtime_) throw std::runtime_error("TensorRT createInferRuntime failed");
    const auto plan = engine_file.plan();
    engine_.reset(runtime_->deserializeCudaEngine(plan.data(), plan.size()));
    if (!engine_) throw std::runtime_error("TensorRT could not deserialize the engine plan");

    int input_count = 0;
    int output_count = 0;
    for (int index = 0; index < engine_->getNbIOTensors(); ++index) {
        const char* name = engine_->getIOTensorName(index);
        if (name == nullptr) throw std::runtime_error("TensorRT returned an unnamed I/O tensor");
        if (engine_->getTensorLocation(name) != nvinfer1::TensorLocation::kDEVICE) {
            throw std::runtime_error("TensorRT I/O tensor '" + std::string(name) + "' must be device-located");
        }
        const int components_per_element = engine_->getTensorComponentsPerElement(name);
        if (engine_->getTensorFormat(name) != nvinfer1::TensorFormat::kLINEAR
            || engine_->getTensorVectorizedDim(name) != -1
            || (components_per_element != -1 && components_per_element != 1)) {
            throw std::runtime_error(
                "TensorRT I/O tensor '" + std::string(name)
                + "' must be linear, non-vectorized, and have one component per element");
        }
        if (engine_->isShapeInferenceIO(name)) {
            throw std::runtime_error("TensorRT shape-inference I/O tensors are not supported: " + std::string(name));
        }
        static_cast<void>(elementSize(engine_->getTensorDataType(name)));
        switch (engine_->getTensorIOMode(name)) {
            case nvinfer1::TensorIOMode::kINPUT:
                ++input_count;
                input_name_ = name;
                break;
            case nvinfer1::TensorIOMode::kOUTPUT:
                ++output_count;
                output_name_ = name;
                break;
            default: throw std::runtime_error("TensorRT tensor has unsupported I/O mode");
        }
    }
    if (input_count != 1 || output_count != 1 || engine_->getNbIOTensors() != 2) {
        throw std::runtime_error("TensorRT engine must expose exactly one input and one output tensor");
    }
    input_type_ = engine_->getTensorDataType(input_name_.c_str());
    output_type_ = engine_->getTensorDataType(output_name_.c_str());

    nvinfer1::Dims input_dimensions = engine_->getTensorShape(input_name_.c_str());
    if (input_dimensions.nbDims != 4) throw std::runtime_error("YOLO input tensor must be rank 4 NCHW");
    const std::array<int, 4> selected_dimensions{
        1,
        3,
        preferred_image_size ? (*preferred_image_size)[0] : 0,
        preferred_image_size ? (*preferred_image_size)[1] : 0,
    };
    for (int index = 0; index < input_dimensions.nbDims; ++index) {
        if (input_dimensions.d[index] == -1) {
            if (selected_dimensions[static_cast<std::size_t>(index)] <= 0) {
                throw std::runtime_error("Dynamic YOLO height or width requires validated imgsz metadata");
            }
            input_dimensions.d[index] = selected_dimensions[static_cast<std::size_t>(index)];
        } else if (input_dimensions.d[index] <= 0) {
            throw std::runtime_error("YOLO input contains an invalid dimension; only -1 may be dynamic");
        } else if (preferred_image_size && index >= 2
            && input_dimensions.d[index] != selected_dimensions[static_cast<std::size_t>(index)]) {
            throw std::runtime_error(
                "TensorRT engine/profile does not support the selected imgsz; select the engine's "
                "fixed size or install an engine with a compatible optimization profile");
        }
    }
    if (input_dimensions.d[0] != 1 || input_dimensions.d[1] != 3
        || input_dimensions.d[2] <= 0 || input_dimensions.d[3] <= 0) {
        throw std::runtime_error("Only fixed batch-1, three-channel NCHW YOLO inputs are supported");
    }
    if (input_dimensions.d[2] > resource_limits::kMaximumImageDimension
        || input_dimensions.d[3] > resource_limits::kMaximumImageDimension) {
        throw std::runtime_error("YOLO input dimensions exceed the supported image limit");
    }
    context_.reset(engine_->createExecutionContext());
    if (!context_) throw std::runtime_error("TensorRT could not create an execution context");
    if (!context_->setInputShape(input_name_.c_str(), input_dimensions)) {
        throw std::runtime_error("TensorRT rejected the selected input shape");
    }
    if (input_dimensions.d[2] > std::numeric_limits<int>::max()
        || input_dimensions.d[3] > std::numeric_limits<int>::max()) {
        throw std::runtime_error("YOLO input dimensions exceed the supported integer range");
    }
    input_height_ = static_cast<int>(input_dimensions.d[2]);
    input_width_ = static_cast<int>(input_dimensions.d[3]);
    input_elements_ = volume(
        input_dimensions, resource_limits::kMaximumInputElements, "TensorRT input");

    const nvinfer1::Dims output_dimensions = context_->getTensorShape(output_name_.c_str());
    output_elements_ = volume(
        output_dimensions, resource_limits::kMaximumOutputElements, "TensorRT output");
    output_shape_.reserve(static_cast<std::size_t>(output_dimensions.nbDims));
    for (int index = 0; index < output_dimensions.nbDims; ++index) {
        output_shape_.push_back(output_dimensions.d[index]);
    }

    input_buffer_ = DeviceBuffer(resource_limits::checkedTensorBytes(
        input_elements_, elementSize(input_type_), "TensorRT input"));
    output_buffer_ = DeviceBuffer(resource_limits::checkedTensorBytes(
        output_elements_, elementSize(output_type_), "TensorRT output"));
    if (input_type_ == nvinfer1::DataType::kFLOAT) host_input_float_.resize(input_elements_);
    else host_input_half_.resize(input_elements_);
    if (output_type_ == nvinfer1::DataType::kFLOAT) host_output_float_.resize(output_elements_);
    else host_output_half_.resize(output_elements_);
    float_output_.resize(output_elements_);
    if (!context_->setTensorAddress(input_name_.c_str(), input_buffer_.data())
        || !context_->setTensorAddress(output_name_.c_str(), output_buffer_.data())) {
        throw std::runtime_error("TensorRT setTensorAddress failed");
    }
}

int TensorRtSession::inputWidth() const noexcept { return input_width_; }
int TensorRtSession::inputHeight() const noexcept { return input_height_; }
const std::vector<std::int64_t>& TensorRtSession::outputShape() const noexcept { return output_shape_; }

InferenceOutput TensorRtSession::infer(std::span<const float> nchw_input) {
    if (nchw_input.size() != input_elements_) {
        throw std::invalid_argument("Preprocessed input length does not match TensorRT input tensor");
    }
    if (input_type_ == nvinfer1::DataType::kFLOAT) {
        std::copy(nchw_input.begin(), nchw_input.end(), host_input_float_.begin());
    } else if (input_type_ == nvinfer1::DataType::kHALF) {
        std::transform(nchw_input.begin(), nchw_input.end(), host_input_half_.begin(), [] (float value) {
            return __float2half(value);
        });
    } else {
        throw std::runtime_error("Unsupported TensorRT input data type");
    }

    const void* host_input = input_type_ == nvinfer1::DataType::kFLOAT
        ? static_cast<const void*>(host_input_float_.data())
        : static_cast<const void*>(host_input_half_.data());
    void* host_output = output_type_ == nvinfer1::DataType::kFLOAT
        ? static_cast<void*>(host_output_float_.data())
        : static_cast<void*>(host_output_half_.data());
    checkCuda(cudaMemcpyAsync(
        input_buffer_.data(), host_input, input_buffer_.size(),
        cudaMemcpyHostToDevice, stream_.get()), "cudaMemcpyAsync input");
    if (!context_->enqueueV3(stream_.get())) {
        throw std::runtime_error("TensorRT enqueueV3 failed");
    }
    checkCuda(cudaMemcpyAsync(
        host_output, output_buffer_.data(), output_buffer_.size(),
        cudaMemcpyDeviceToHost, stream_.get()), "cudaMemcpyAsync output");
    checkCuda(cudaStreamSynchronize(stream_.get()), "cudaStreamSynchronize");

    if (output_type_ == nvinfer1::DataType::kFLOAT) {
        std::copy(host_output_float_.begin(), host_output_float_.end(), float_output_.begin());
    } else if (output_type_ == nvinfer1::DataType::kHALF) {
        std::transform(host_output_half_.begin(), host_output_half_.end(), float_output_.begin(), [] (__half value) {
            return __half2float(value);
        });
    } else {
        throw std::runtime_error("Unsupported TensorRT output data type");
    }
    return {float_output_, output_shape_};
}

DeviceSummary selectCudaDevice(std::optional<int> requested_device) {
    int count{};
    checkCuda(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    std::vector<CudaDeviceInfo> devices;
    devices.reserve(static_cast<std::size_t>(std::max(0, count)));
    for (int index = 0; index < count; ++index) {
        cudaDeviceProp properties{};
        checkCuda(cudaGetDeviceProperties(&properties, index), "cudaGetDeviceProperties");
        devices.push_back({index, properties.name, properties.major, properties.minor});
    }
    const int device = selectCompatibleCudaDevice(devices, requested_device);
    cudaDeviceProp properties{};
    checkCuda(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
    checkCuda(cudaSetDevice(device), "cudaSetDevice");
    return {
        count,
        device,
        properties.name,
        properties.totalGlobalMem,
        properties.major,
        properties.minor,
    };
}

}  // namespace cuajone
