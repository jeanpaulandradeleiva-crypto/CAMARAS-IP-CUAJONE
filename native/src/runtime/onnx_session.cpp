// SPDX-License-Identifier: AGPL-3.0-only

#include "cuajone/onnx_session.hpp"
#include "cuajone/resource_limits.hpp"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace cuajone {
struct OnnxSession::Impl {
    Impl(
        const std::filesystem::path& model_path,
        ModelRole expected_role,
        const OnnxSessionOptions& options,
        std::optional<int> selected_image_size)
        : environment(ORT_LOGGING_LEVEL_WARNING, "NexoAIVision"),
          session_options(),
          session(nullptr),
          verified_model(verifyOnnxModel(model_path, expected_role)) {
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        session_options.DisableMemPattern();
        if (options.execution_provider == OnnxExecutionProvider::Cuda) {
            if (!options.cuda_device) {
                throw std::invalid_argument("ONNX CUDA execution provider requires a CUDA device index");
            }
            OrtCUDAProviderOptions provider_options{};
            provider_options.device_id = *options.cuda_device;
            session_options.AppendExecutionProvider_CUDA(provider_options);
        }
        session = Ort::Session(
            environment, verified_model.bytes.data(), verified_model.bytes.size(), session_options);
        if (session.GetInputCount() != 1 || session.GetOutputCount() != 1) {
            throw std::runtime_error("ONNX model must expose exactly one input and one output tensor");
        }
        input_name = session.GetInputNameAllocated(0, allocator).get();
        output_name = session.GetOutputNameAllocated(0, allocator).get();
        const auto input_type_info = session.GetInputTypeInfo(0);
        const auto output_type_info = session.GetOutputTypeInfo(0);
        const auto input_info = input_type_info.GetTensorTypeAndShapeInfo();
        const auto output_info = output_type_info.GetTensorTypeAndShapeInfo();
        if (input_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
            || output_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            throw std::runtime_error("ONNX models must use float32 input and output tensors");
        }
        input_shape = input_info.GetShape();
        const auto declared_output_shape = output_info.GetShape();
        const auto& contract = verified_model.manifest;
        if (input_name != contract.input.name || output_name != contract.output.name
            || input_shape != contract.input.shape || declared_output_shape != contract.output.shape) {
            throw std::runtime_error("ONNX Runtime I/O does not exactly match the approved manifest");
        }
        if (contract.dynamicShapes()) {
            const int concrete_size = selected_image_size.value_or(kDefaultImageSize);
            validateImageSize(concrete_size);
            if (std::ranges::find(contract.allowed_image_sizes, concrete_size)
                == contract.allowed_image_sizes.end()) {
                throw std::runtime_error("Selected imgsz is not allowed by the dynamic ONNX manifest");
            }
            input_shape = {1, 3, concrete_size, concrete_size};
            input_height = concrete_size;
            input_width = concrete_size;
            if (expected_role == ModelRole::Ppe) {
                output_shape = {
                    1, 12,
                    static_cast<std::int64_t>(yoloPredictionCount(concrete_size)),
                };
            } else {
                output_shape = contract.output.shape;
            }
        } else {
            if (input_shape.size() != 4 || input_shape[0] != 1 || input_shape[1] != 3
                || input_shape[2] <= 0 || input_shape[3] <= 0) {
                throw std::runtime_error("ONNX input must be fixed batch-1, three-channel NCHW");
            }
            if (selected_image_size
                && (input_shape[2] != *selected_image_size || input_shape[3] != *selected_image_size)) {
                throw std::runtime_error(
                    "Static ONNX model does not support the selected imgsz; install the managed dynamic model set");
            }
            if (input_shape[2] > kMaximumImageSize || input_shape[3] > kMaximumImageSize) {
                throw std::runtime_error("ONNX input dimensions exceed the supported image limit");
            }
            input_height = static_cast<int>(input_shape[2]);
            input_width = static_cast<int>(input_shape[3]);
            output_shape = declared_output_shape;
        }
        input_elements = resource_limits::checkedVolume(
            input_shape, resource_limits::kMaximumInputElements, "ONNX input");
        static_cast<void>(resource_limits::checkedTensorBytes(
            input_elements, sizeof(float), "ONNX input"));
        const std::size_t output_elements = resource_limits::checkedVolume(
            output_shape, resource_limits::kMaximumOutputElements, "ONNX output");
        static_cast<void>(resource_limits::checkedTensorBytes(
            output_elements, sizeof(float), "ONNX output"));
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
        if (actual_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            throw std::runtime_error("ONNX Runtime output type changed after session validation");
        }
        const std::size_t count = resource_limits::checkedVolume(
            actual_shape, resource_limits::kMaximumOutputElements, "ONNX Runtime output");
        static_cast<void>(resource_limits::checkedTensorBytes(
            count, sizeof(float), "ONNX Runtime output"));
        if (actual_shape != output_shape || actual_info.GetElementCount() != count) {
            throw std::runtime_error("ONNX Runtime output shape or type changed after session validation");
        }
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

OnnxSession::OnnxSession(
    const std::filesystem::path& model_path,
    ModelRole expected_role,
    OnnxSessionOptions options,
    std::optional<int> image_size)
    : impl_(std::make_unique<Impl>(model_path, expected_role, options, image_size)) {}

OnnxSession::~OnnxSession() = default;
int OnnxSession::inputWidth() const noexcept { return impl_->input_width; }
int OnnxSession::inputHeight() const noexcept { return impl_->input_height; }
const std::vector<std::int64_t>& OnnxSession::outputShape() const noexcept { return impl_->output_shape; }
const OnnxModelManifest& OnnxSession::manifest() const noexcept { return impl_->verified_model.manifest; }
InferenceOutput OnnxSession::infer(std::span<const float> input) { return impl_->run(input); }

}  // namespace cuajone
