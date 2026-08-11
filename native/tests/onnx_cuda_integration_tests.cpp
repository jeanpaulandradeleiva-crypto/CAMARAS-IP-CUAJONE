// SPDX-License-Identifier: AGPL-3.0-only

#define NOMINMAX

#include "cuajone/compute.hpp"
#include "cuajone/contracts.hpp"
#include "cuajone/engine_pipeline.hpp"
#include "cuajone/onnx_session.hpp"
#include "cuajone/yolo_decode.hpp"

#include "onnx_fixture.hpp"

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <numeric>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr int kSkipped = 77;

using namespace cuajone;
using namespace cuajone::test;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::string windowsError(DWORD code) {
    LPSTR message{};
    const DWORD size = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM
            | FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr, code, 0, reinterpret_cast<LPSTR>(&message), 0, nullptr);
    std::string result = size == 0 ? "Windows error " + std::to_string(code) : std::string(message, size);
    if (message != nullptr) LocalFree(message);
    while (!result.empty() && (result.back() == '\r' || result.back() == '\n')) result.pop_back();
    return result;
}

std::filesystem::path cudaRuntimeDirectory() {
    const DWORD required = GetEnvironmentVariableW(L"CUAJONE_ONNX_CUDA_RUNTIME_DIR", nullptr, 0);
    if (required == 0) {
        throw std::runtime_error(
            "CUDA provider runtime directory is not configured; set CUAJONE_ONNX_CUDA_RUNTIME_DIR "
            "to the directory containing onnxruntime_providers_cuda.dll and its DLL closure");
    }
    std::wstring value(required, L'\0');
    const DWORD written = GetEnvironmentVariableW(
        L"CUAJONE_ONNX_CUDA_RUNTIME_DIR", value.data(), static_cast<DWORD>(value.size()));
    if (written == 0 || written >= value.size()) {
        throw std::runtime_error("Could not read CUAJONE_ONNX_CUDA_RUNTIME_DIR");
    }
    value.resize(written);
    return value;
}

void configureProviderDllSearch(const std::filesystem::path& runtime_directory) {
    const auto provider = runtime_directory / L"onnxruntime_providers_cuda.dll";
    const auto shared = runtime_directory / L"onnxruntime_providers_shared.dll";
    require(std::filesystem::is_regular_file(provider),
        "CUDA provider library is missing: " + provider.string());
    require(std::filesystem::is_regular_file(shared),
        "CUDA provider support library is missing: " + shared.string());
    if (!SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS)) {
        throw std::runtime_error("Could not configure the Windows DLL search policy: "
            + windowsError(GetLastError()));
    }
    if (AddDllDirectory(runtime_directory.c_str()) == nullptr) {
        throw std::runtime_error("Could not add the CUDA provider runtime directory to the DLL search path: "
            + runtime_directory.string() + "; " + windowsError(GetLastError()));
    }
}

std::string readText(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Could not read ONNX Runtime profile: " + path.string());
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

std::map<int, std::string> stagedPpeLabels() {
    return {
        {0, "Gloves"},
        {1, "Person"},
        {2, "Safety_boots"},
        {3, "Vest"},
        {4, "respirador"},
        {5, "tapaorejas"},
        {6, "Hard_hat"},
        {7, "lentes_protectores"},
    };
}

void verifyCudaProfile(
    const std::filesystem::path& model_path,
    const std::filesystem::path& profile_prefix,
    int device) {
    Ort::Env environment(ORT_LOGGING_LEVEL_WARNING, "cuajone_onnx_cuda_integration");
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    options.DisableMemPattern();
    options.EnableProfiling(profile_prefix.c_str());
    OrtCUDAProviderOptions provider_options{};
    provider_options.device_id = device;
    options.AppendExecutionProvider_CUDA(provider_options);
    Ort::Session session(environment, model_path.c_str(), options);

    require(session.GetInputCount() == 1 && session.GetOutputCount() == 1,
        "Staged PPE model must expose exactly one input and one output");
    const auto input_type_info = session.GetInputTypeInfo(0);
    const auto input_info = input_type_info.GetTensorTypeAndShapeInfo();
    const auto declared_shape = input_info.GetShape();
    require(input_info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
            && declared_shape.size() == 4 && declared_shape[0] == 1 && declared_shape[1] == 3,
        "Staged PPE model input is not batch-1 FP32 NCHW");
    const bool static_spatial = declared_shape[2] > 0 && declared_shape[3] > 0;
    const bool dynamic_spatial = declared_shape[2] == -1 && declared_shape[3] == -1;
    require(static_spatial || dynamic_spatial,
        "Staged PPE model input must use fixed or bounded dynamic spatial dimensions");
    auto shape = declared_shape;
    if (dynamic_spatial) {
        shape[2] = 640;
        shape[3] = 640;
    }
    const std::size_t input_elements = std::accumulate(
        shape.begin(), shape.end(), std::size_t{1},
        [](std::size_t product, std::int64_t dimension) {
            return product * static_cast<std::size_t>(dimension);
        });
    std::vector<float> input(input_elements, 0.25F);
    const Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory, const_cast<float*>(input.data()), input.size(), shape.data(), shape.size());
    Ort::AllocatorWithDefaultOptions allocator;
    const auto input_name = session.GetInputNameAllocated(0, allocator);
    const auto output_name = session.GetOutputNameAllocated(0, allocator);
    const char* input_names[]{input_name.get()};
    const char* output_names[]{output_name.get()};
    const auto output = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
    require(output.size() == 1 && output.front().IsTensor()
            && output.front().GetTensorTypeAndShapeInfo().GetElementCount() > 0,
        "Staged PPE CUDA profile session returned no tensor output");

    const auto profile_path = std::filesystem::path(session.EndProfilingAllocated(allocator).get());
    const std::string profile = readText(profile_path);
    require(std::regex_search(profile, std::regex(R"("provider"\s*:\s*"CUDAExecutionProvider")")),
        "CUDAExecutionProvider did not execute the staged PPE graph. Verify the provider DLL search path and CUDA runtime closure");
}

void verifyPpeOnnxSessionContract(const std::filesystem::path& ppe_model, int device) {
    const OnnxSessionOptions options{OnnxExecutionProvider::Cuda, device};
    OnnxSession baseline(ppe_model, ModelRole::Ppe, options);
    const auto& manifest = baseline.manifest();
    const std::vector<int> image_sizes = manifest.dynamicShapes()
        ? manifest.allowed_image_sizes
        : std::vector<int>{0};
    for (const int image_size : image_sizes) {
        OnnxSession session(
            ppe_model,
            ModelRole::Ppe,
            options,
            manifest.dynamicShapes() ? std::optional<int>{image_size} : std::nullopt);
        if (manifest.dynamicShapes()) {
            require(session.inputWidth() == image_size && session.inputHeight() == image_size,
                "OnnxSession did not resolve the approved PPE input dimensions");
        }
        std::vector<float> input(
            static_cast<std::size_t>(session.inputWidth())
                * static_cast<std::size_t>(session.inputHeight()) * 3,
            0.25F);
        const InferenceOutput output = session.infer(input);
        const std::vector<std::int64_t> expected_shape = manifest.dynamicShapes()
            ? std::vector<std::int64_t>{1, 12, static_cast<std::int64_t>(yoloPredictionCount(image_size))}
            : session.outputShape();
        require(std::ranges::equal(output.shape, expected_shape),
            "OnnxSession returned an unexpected PPE output shape");
    }
}

void verifyStandalonePose(
    const std::filesystem::path& pose_model,
    OnnxSessionOptions options,
    std::string_view provider_name) {
    std::cout << "INFO: constructing standalone " << provider_name
              << " OnnxSession with pose model: "
              << pose_model.string() << std::endl;
    OnnxSession pose_session(pose_model, ModelRole::Pose, options);
    std::vector<float> input(
        static_cast<std::size_t>(pose_session.inputWidth())
            * static_cast<std::size_t>(pose_session.inputHeight()) * 3,
        0.25F);
    std::cout << "INFO: running standalone pose " << provider_name << " inference" << std::endl;
    const InferenceOutput output = pose_session.infer(input);
    const YoloSchema schema = validatePoseSchema(output.shape, 1, 17, 3);
    require(output.values.size() == schema.predictions * schema.channels,
        "Staged pose model returned an unexpected output element count");
    std::cout << "PASS: standalone pose " << provider_name
              << " inference returned a valid " << schema.predictions << 'x'
              << schema.channels << " pose tensor" << std::endl;
}

void verifyNativePipeline(
    const std::filesystem::path& ppe_model,
    const std::filesystem::path& pose_model,
    const std::filesystem::path& person_image,
    int device) {
    EnginePipelineConfig config;
    config.backend = ComputeBackend::Cuda;
    config.provider = InferenceProvider::OnnxRuntimeCuda;
    config.ppe_onnx = ppe_model;
    config.pose_onnx = pose_model;
    config.ppe_labels = stagedPpeLabels();
    config.pose_class_count = 1;
    config.pose_keypoint_shape = {17, 3};
    config.device = device;
    config.analytics.mode = AnalyticsMode::PpeFall;

    std::cout << "INFO: constructing NativeEnginePipeline with staged PPE and pose ONNX models"
              << std::endl;
    NativeEnginePipeline pipeline(std::move(config));
    const auto& summary = pipeline.summary();
    require(summary.backend == ComputeBackend::Cuda,
        "NativeEnginePipeline did not report the CUDA backend");
    require(summary.provider
            == "ONNX Runtime CUDAExecutionProvider (PPE) + CPUExecutionProvider (pose)",
        "NativeEnginePipeline did not report the PPE-CUDA/pose-CPU hybrid provider split");
    require(summary.pose_loaded, "NativeEnginePipeline did not load the pose model in ppe-fall mode");

    std::cout << "INFO: processing offline person image: " << person_image.string() << std::endl;
    cv::Mat frame = cv::imread(person_image.string(), cv::IMREAD_COLOR);
    require(!frame.empty() && frame.type() == CV_8UC3,
        "Could not load the offline person regression image");
    const ProcessedFrame processed = pipeline.processFrame(
        frame, "cuda-integration", 1, 1000, "2026-08-01T00:00:00Z");
    require(processed.canonical.source_id == "cuda-integration"
            && processed.canonical.frame_id == 1
            && processed.canonical.monotonic_timestamp_ms == 1000
            && processed.canonical.frame_width == frame.cols
            && processed.canonical.frame_height == frame.rows,
        "NativeEnginePipeline returned invalid canonical frame metadata");
    const std::string canonical = canonicalJson(processed.canonical);
    require(!canonical.empty()
            && canonical.find(R"("contract_version":"1.0.0")") != std::string::npos
            && canonical.find(R"("source_id":"cuda-integration")") != std::string::npos,
        "NativeEnginePipeline did not serialize valid canonical output");
    require(!processed.canonical.people.empty(),
        "Real staged pose model produced no native Person output for the known person image");
    require(std::all_of(
                processed.canonical.people.begin(), processed.canonical.people.end(),
                [](const auto& person) { return person.keypoints.size() == 17; }),
        "Real staged pose output did not preserve all 17 keypoints");
    std::cout << "PASS: NativeEnginePipeline ppe-fall processed a deterministic BGR frame with "
              << summary.provider << "; people=" << processed.canonical.people.size()
              << "; pose loaded; canonical bytes=" << canonical.size() << '\n';
}

int cudaDeviceOrSkip() {
    const auto probe = probeHardware();
    if (probe.status != HardwareProbeStatus::CudaReady) {
        std::cout << "SKIP: no compatible NVIDIA GPU/CUDA driver: "
                  << hardwareProbeStatusName(probe.status) << "; " << probe.detail << '\n';
        throw kSkipped;
    }
    return selectCompatibleCudaDevice(probe.cuda_devices, std::nullopt);
}

void runPpeOnly(const std::filesystem::path& ppe_model) {
    const int device = cudaDeviceOrSkip();
    configureProviderDllSearch(cudaRuntimeDirectory());

    TemporaryDirectory directory;
    verifyCudaProfile(
        ppe_model, directory.path() / L"cuajone_staged_ppe_cuda_profile", device);
    verifyPpeOnnxSessionContract(ppe_model, device);

    EnginePipelineConfig config;
    config.backend = ComputeBackend::Cuda;
    config.provider = InferenceProvider::OnnxRuntimeCuda;
    config.ppe_onnx = ppe_model;
    config.ppe_labels = stagedPpeLabels();
    config.device = device;
    config.analytics.mode = AnalyticsMode::PpeOnly;
    NativeEnginePipeline pipeline(std::move(config));
    require(pipeline.summary().backend == ComputeBackend::Cuda
            && pipeline.summary().provider == "ONNX Runtime CUDAExecutionProvider"
            && !pipeline.summary().pose_loaded,
        "PPE-only production pipeline did not preserve its CUDA-only provider contract");
    cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(32, 64, 96));
    const ProcessedFrame processed = pipeline.processFrame(
        frame, "ppe-cuda-integration", 1, 1000, "2026-08-01T00:00:00Z");
    require(processed.canonical.source_id == "ppe-cuda-integration"
            && processed.canonical.frame_width == frame.cols
            && processed.canonical.frame_height == frame.rows,
        "PPE-only production pipeline returned invalid canonical frame metadata");
    std::cout << "PASS: CUDAExecutionProvider executed the real staged PPE graph, OnnxSession "
                 "validated its static v1 or all bounded dynamic v2 input sizes, and the PPE-only "
                 "production pipeline processed a deterministic BGR frame"
              << std::endl;
}

void runDynamicPpe(const std::filesystem::path& ppe_model, int image_size) {
    const int device = cudaDeviceOrSkip();
    configureProviderDllSearch(cudaRuntimeDirectory());
    OnnxSession session(
        ppe_model,
        ModelRole::Ppe,
        OnnxSessionOptions{OnnxExecutionProvider::Cuda, device},
        image_size);
    std::vector<float> input(
        static_cast<std::size_t>(session.inputWidth())
            * static_cast<std::size_t>(session.inputHeight()) * 3,
        0.25F);
    const auto started = std::chrono::steady_clock::now();
    const InferenceOutput output = session.infer(input);
    const auto elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    const std::vector<std::int64_t> expected_shape{
        1, 12, static_cast<std::int64_t>(yoloPredictionCount(image_size))};
    require(std::ranges::equal(output.shape, expected_shape)
            && output.values.size() == static_cast<std::size_t>(12 * yoloPredictionCount(image_size)),
        "Dynamic PPE CUDA session returned an unexpected output shape");
    std::cout << "PASS: dynamic PPE OnnxSession CUDA inference at imgsz " << image_size
              << " returned " << output.shape[2] << " predictions in " << elapsed << " ms"
              << std::endl;
}

void runStandalonePoseCpu(const std::filesystem::path& pose_model) {
    verifyStandalonePose(pose_model, OnnxSessionOptions{}, "CPU");
}

void runPipeline(
    const std::filesystem::path& ppe_model,
    const std::filesystem::path& pose_model,
    const std::filesystem::path& person_image) {
    const int device = cudaDeviceOrSkip();
    configureProviderDllSearch(cudaRuntimeDirectory());
    const auto probe = probeHardware();
    const auto selected = std::find_if(probe.cuda_devices.begin(), probe.cuda_devices.end(), [&](const auto& value) {
        return value.device_index == device;
    });
    require(selected != probe.cuda_devices.end(), "Selected CUDA device disappeared from the hardware probe");
    std::cout << "INFO: hybrid pipeline selected GPU " << selected->name
              << " (device " << device << ", SM " << selected->compute_major << '.' << selected->compute_minor
              << ')' << std::endl;
    verifyNativePipeline(ppe_model, pose_model, person_image, device);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 3 && std::string_view(argv[1]) == "standalone-pose-cpu") {
            runStandalonePoseCpu(argv[2]);
        } else if (argc == 3 && std::string_view(argv[1]) == "ppe-only") {
            runPpeOnly(argv[2]);
        } else if (argc == 4 && std::string_view(argv[1]) == "dynamic-ppe") {
            runDynamicPpe(argv[2], std::stoi(argv[3]));
        } else if (argc == 5 && std::string_view(argv[1]) == "pipeline") {
            runPipeline(argv[2], argv[3], argv[4]);
        } else {
            throw std::invalid_argument("Usage: cuajone_onnx_cuda_integration_tests "
                "standalone-pose-cpu <pose.onnx> | ppe-only <ppe.onnx> | "
                "dynamic-ppe <ppe.onnx> <imgsz> | "
                "pipeline <ppe.onnx> <pose.onnx> <person-image>");
        }
        return 0;
    } catch (int code) {
        return code;
    } catch (const Ort::Exception& error) {
        std::cerr << "FAIL: ONNX Runtime session creation or inference failed: "
                  << error.what() << '\n';
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
    }
    return 1;
}
