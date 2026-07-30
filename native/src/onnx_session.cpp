// SPDX-License-Identifier: AGPL-3.0-only

#include "cuajone/onnx_session.hpp"
#include "cuajone/resource_limits.hpp"

#include <onnxruntime_cxx_api.h>

#include <stdexcept>
#include <string>
#include <utility>

namespace cuajone {
struct OnnxCpuSession::Impl {
    Impl(const std::filesystem::path& model_path, ModelRole expected_role)
        : environment(ORT_LOGGING_LEVEL_WARNING, "cuajone_native"),
          session_options(),
          session(nullptr),
          verified_model(verifyOnnxModel(model_path, expected_role)) {
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        session_options.DisableMemPattern();
        session = Ort::Session(
            environment, verified_model.bytes.data(), verified_model.bytes.size(), session_options);
        if (session.GetInputCount() != 1 || session.GetOutputCount() != 1) {
            throw std::runtime_error("ONNX model must expose exactly one input and one output tensor");
        }
        input_name = session.GetInputNameAllocated(0, allocator).get();
        output_name = session.GetOutputNameAllocated(0, allocator).get();
        const auto input_info = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
        const auto output_info = session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo();
        if (input_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
            || output_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            throw std::runtime_error("ONNX CPU models must use float32 input and output tensors");
        }
        input_shape = input_info.GetShape();
        output_shape = output_info.GetShape();
        if (input_shape.size() != 4 || input_shape[0] != 1 || input_shape[1] != 3
            || input_shape[2] <= 0 || input_shape[3] <= 0) {
            throw std::runtime_error("ONNX input must be fixed batch-1, three-channel NCHW");
        }
        if (input_shape[2] > resource_limits::kMaximumImageDimension
            || input_shape[3] > resource_limits::kMaximumImageDimension) {
            throw std::runtime_error("ONNX input dimensions exceed the supported image limit");
        }
        input_height = static_cast<int>(input_shape[2]);
        input_width = static_cast<int>(input_shape[3]);
        input_elements = resource_limits::checkedVolume(
            input_shape, resource_limits::kMaximumInputElements, "ONNX input");
        static_cast<void>(resource_limits::checkedTensorBytes(
            input_elements, sizeof(float), "ONNX input"));
        const std::size_t output_elements = resource_limits::checkedVolume(
            output_shape, resource_limits::kMaximumOutputElements, "ONNX output");
        static_cast<void>(resource_limits::checkedTensorBytes(
            output_elements, sizeof(float), "ONNX output"));
        const auto& contract = verified_model.manifest;
        if (input_name != contract.input.name || output_name != contract.output.name
            || input_shape != contract.input.shape || output_shape != contract.output.shape) {
            throw std::runtime_error("ONNX Runtime I/O does not exactly match the approved manifest");
        }
    }

    InferenceOutput run(std::span<const float> input) {
        if (input.size() != input_elements) {
            throw std::invalid_argument("Preprocessed input length does not match ONNX input tensor");
        }
        const Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory, const_cast<float*>(input.data()), input.size(), input_shape.data(), input_shape.size());
        const char* input_names[]{input_name.c_str()};
        const char* output_names[]{output_name.c_str()};
        auto outputs = session.Run(
            Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
        if (outputs.size() != 1 || !outputs.front().IsTensor()) {
            throw std::runtime_error("ONNX Runtime returned an invalid output collection");
        }
        const auto actual_info = outputs.front().GetTensorTypeAndShapeInfo();
        const auto actual_shape = actual_info.GetShape();
        if (actual_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
            || actual_shape != output_shape) {
            throw std::runtime_error("ONNX Runtime output shape or type changed after session validation");
        }
        const std::size_t count = actual_info.GetElementCount();
        const float* values = outputs.front().GetTensorData<float>();
        output_values.assign(values, values + count);
        return {output_values, output_shape};
    }

    Ort::Env environment;
    Ort::SessionOptions session_options;
    Ort::Session session;
    VerifiedOnnxModel verified_model;
    Ort::AllocatorWithDefaultOptions allocator;
    std::string input_name;
    std::string output_name;
    std::vector<std::int64_t> input_shape;
    std::vector<std::int64_t> output_shape;
    std::vector<float> output_values;
    int input_width{};
    int input_height{};
    std::size_t input_elements{};
};

OnnxCpuSession::OnnxCpuSession(const std::filesystem::path& model_path, ModelRole expected_role)
    : impl_(std::make_unique<Impl>(model_path, expected_role)) {}

OnnxCpuSession::~OnnxCpuSession() = default;
int OnnxCpuSession::inputWidth() const noexcept { return impl_->input_width; }
int OnnxCpuSession::inputHeight() const noexcept { return impl_->input_height; }
const std::vector<std::int64_t>& OnnxCpuSession::outputShape() const noexcept { return impl_->output_shape; }
InferenceOutput OnnxCpuSession::infer(std::span<const float> input) { return impl_->run(input); }

}  // namespace cuajone
