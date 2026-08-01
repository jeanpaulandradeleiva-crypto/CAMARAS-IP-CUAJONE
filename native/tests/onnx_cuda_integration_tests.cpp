// SPDX-License-Identifier: AGPL-3.0-only

#include "cuajone/compute.hpp"
#include "cuajone/onnx_session.hpp"

#include "onnx_fixture.hpp"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>
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

void verifyCudaProfile(
    const std::filesystem::path& model_path,
    const std::filesystem::path& profile_prefix,
    int device,
    const std::vector<float>& input) {
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

    const std::array<std::int64_t, 4> shape{1, 3, 2, 2};
    const Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory, const_cast<float*>(input.data()), input.size(), shape.data(), shape.size());
    const char* input_names[]{"input"};
    const char* output_names[]{"output"};
    const auto output = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
    require(output.size() == 1 && output.front().IsTensor(), "CUDA ONNX profile session returned no tensor output");
    const float* values = output.front().GetTensorData<float>();
    for (std::size_t index = 0; index < input.size(); ++index) {
        require(values[index] == input[index] + 1.0F, "CUDA ONNX profile session returned an incorrect Add result");
    }

    Ort::AllocatorWithDefaultOptions allocator;
    const auto profile_path = std::filesystem::path(session.EndProfilingAllocated(allocator).get());
    const std::string profile = readText(profile_path);
    require(std::regex_search(profile, std::regex(R"("provider"\s*:\s*"CUDAExecutionProvider")")),
        "CUDAExecutionProvider did not execute the Add node. Verify the provider DLL search path and CUDA runtime closure");
}

void run() {
    const auto probe = probeHardware();
    if (probe.status != HardwareProbeStatus::CudaReady) {
        std::cout << "SKIP: no compatible NVIDIA GPU/CUDA driver: "
                  << hardwareProbeStatusName(probe.status) << "; " << probe.detail << '\n';
        throw kSkipped;
    }
    const int device = selectCompatibleCudaDevice(probe.cuda_devices, std::nullopt);
    configureProviderDllSearch(cudaRuntimeDirectory());

    TemporaryDirectory directory;
    const Bytes model = addModel();
    const auto model_path = writeModelSet(directory, "cuda-add.onnx", model);
    const std::vector<float> input{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

    OnnxSession project_session(model_path, ModelRole::Ppe, OnnxSessionOptions{
        OnnxExecutionProvider::Cuda, device,
    });
    const InferenceOutput output = project_session.infer(input);
    require(output.values.size() == input.size(), "Project CUDA ONNX session changed the output size");
    for (std::size_t index = 0; index < input.size(); ++index) {
        require(output.values[index] == input[index] + 1.0F,
            "Project CUDA ONNX session returned an incorrect Add result");
    }

    verifyCudaProfile(model_path, directory.path() / L"cuajone_onnx_cuda_profile", device, input);
    const auto selected = std::find_if(probe.cuda_devices.begin(), probe.cuda_devices.end(), [&](const auto& value) {
        return value.device_index == device;
    });
    require(selected != probe.cuda_devices.end(), "Selected CUDA device disappeared from the hardware probe");
    std::cout << "PASS: CUDAExecutionProvider executed Add on GPU " << selected->name
              << " (device " << device << ", SM " << selected->compute_major << '.' << selected->compute_minor
              << ")\n";
}

}  // namespace

int main() {
    try {
        run();
        return 0;
    } catch (int code) {
        return code;
    } catch (const Ort::Exception& error) {
        std::cerr << "FAIL: CUDAExecutionProvider session creation or inference failed. "
                  << "Verify the provider DLL search path and CUDA runtime closure: " << error.what() << '\n';
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
    }
    return 1;
}
