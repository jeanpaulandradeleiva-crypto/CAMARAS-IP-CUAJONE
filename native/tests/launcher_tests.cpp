// SPDX-License-Identifier: AGPL-3.0-only

#include "cuajone/launcher_support.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace cuajone::launcher;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename Function>
void requireThrows(Function function, const std::string& message) {
    try {
        function();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error(message);
}

bool contains(const std::vector<std::wstring>& arguments, std::wstring_view value) {
    for (const auto& argument : arguments) {
        if (argument == value) return true;
    }
    return false;
}

class TemporaryTree {
public:
    TemporaryTree() {
        root_ = std::filesystem::temp_directory_path()
            / (L"cuajone-launcher-tests-" + std::to_wstring(
                std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(root_ / L"output");
    }

    ~TemporaryTree() {
        std::error_code error;
        std::filesystem::remove_all(root_, error);
    }

    std::filesystem::path makeFile(std::wstring_view name) const {
        const auto path = root_ / name;
        std::ofstream(path, std::ios::binary).put('x');
        return path;
    }

    const std::filesystem::path& root() const { return root_; }

private:
    std::filesystem::path root_;
};

LauncherSettings baseSettings(const TemporaryTree& tree) {
    LauncherSettings settings;
    settings.source = L"rtsp://camera.example:554/axis-media/media.amp";
    settings.output = tree.root() / L"output";
    settings.ppe_labels = L"Person,Hard_hat,Vest";
    return settings;
}

void addTensorRtModels(LauncherSettings& settings, const TemporaryTree& tree) {
    settings.ppe_engine = tree.makeFile(L"ppe.engine");
    settings.pose_engine = tree.makeFile(L"pose.engine");
}

void addOnnxModels(LauncherSettings& settings, const TemporaryTree& tree) {
    settings.ppe_onnx = tree.makeFile(L"ppe.onnx");
    tree.makeFile(L"ppe.onnx.manifest.json");
    settings.pose_onnx = tree.makeFile(L"pose.onnx");
    tree.makeFile(L"pose.onnx.manifest.json");
}

void testRtspCameraSourceIsRequired() {
    TemporaryTree tree;
    auto settings = baseSettings(tree);
    addTensorRtModels(settings, tree);
    settings.source = tree.makeFile(L"source with space.mp4").wstring();
    requireThrows([&] { buildLaunchPlan(settings, false); },
        "Launcher accepted a local media file as the camera source");
    settings.source = L"rtsp://";
    requireThrows([&] { buildLaunchPlan(settings, false); },
        "Launcher accepted an RTSP URL without a host");
    settings.source = L"rtsp://camera.example/live";
    const auto plan = buildLaunchPlan(settings, false);
    require(contains(plan.arguments, L"--source")
            && contains(plan.arguments, L"rtsp://camera.example/live"),
        "Launcher did not emit the RTSP camera URL");
}

void testModelMatrixAndArguments() {
    TemporaryTree tree;
    auto settings = baseSettings(tree);
    settings.compute_mode = ComputeMode::Auto;
    settings.source_label = L"CAM_CUAJONE_01";
    settings.runtime_options = {{L"--target-fps", L"12"}, {L"--ppe-conf", L"0.48"}};
    requireThrows([&] { buildLaunchPlan(settings, false); },
        "Auto accepted no model candidate");

    settings.ppe_engine = tree.makeFile(L"ppe.plan");
    settings.pose_engine = tree.makeFile(L"pose.plan");
    requireThrows([&] { buildLaunchPlan(settings, false); },
        "Auto accepted CUDA files with unsupported extensions");

    addTensorRtModels(settings, tree);
    const auto cuda_auto = buildLaunchPlan(settings, true);
    require(cuda_auto.has_cuda_candidate && !cuda_auto.has_cpu_candidate,
        "Auto did not recognize the CUDA candidate");
    require(contains(cuda_auto.arguments, L"--preflight")
            && contains(cuda_auto.arguments, L"--ppe-engine")
            && contains(cuda_auto.arguments, L"--pose-engine")
            && contains(cuda_auto.arguments, L"--source-label")
            && contains(cuda_auto.arguments, L"CAM_CUAJONE_01")
            && contains(cuda_auto.arguments, L"--target-fps")
            && contains(cuda_auto.arguments, L"12")
            && !contains(cuda_auto.arguments, L"--ppe-onnx"),
        "CUDA Auto plan emitted the wrong arguments");

    settings.compute_mode = ComputeMode::Cuda;
    const auto cuda = buildLaunchPlan(settings, false);
    require(cuda.has_cuda_candidate && contains(cuda.arguments, L"--ppe-engine")
            && contains(cuda.arguments, L"--pose-engine")
            && !contains(cuda.arguments, L"--ppe-onnx"),
        "Explicit CUDA plan emitted the wrong model family");

    settings.compute_mode = ComputeMode::Cpu;
    requireThrows([&] { buildLaunchPlan(settings, false); },
        "CPU accepted missing ONNX models");
    addOnnxModels(settings, tree);
    const auto cpu = buildLaunchPlan(settings, false);
    require(cpu.has_cpu_candidate && contains(cpu.arguments, L"--ppe-onnx")
            && contains(cpu.arguments, L"--pose-onnx")
            && !contains(cpu.arguments, L"--ppe-engine"),
        "CPU plan emitted the wrong model family");

    settings.ppe_engine.clear();
    settings.pose_engine.clear();
    settings.compute_mode = ComputeMode::Cuda;
    const auto cuda_onnx = buildLaunchPlan(settings, false);
    require(cuda_onnx.has_cuda_candidate && contains(cuda_onnx.arguments, L"--ppe-onnx")
            && contains(cuda_onnx.arguments, L"--pose-onnx")
            && !contains(cuda_onnx.arguments, L"--ppe-engine"),
        "CUDA plan rejected the validated ONNX model family");

    settings.ppe_onnx = tree.makeFile(L"ppe.pb");
    tree.makeFile(L"ppe.pb.manifest.json");
    requireThrows([&] { buildLaunchPlan(settings, false); },
        "CPU accepted a model with a non-ONNX extension");
    settings.ppe_onnx = tree.root() / L"ppe.onnx";

    addTensorRtModels(settings, tree);
    settings.compute_mode = ComputeMode::Auto;
    const auto both = buildLaunchPlan(settings, false);
    require(both.has_cuda_candidate && both.has_cpu_candidate
            && contains(both.arguments, L"--ppe-engine")
            && contains(both.arguments, L"--ppe-onnx"),
        "Auto did not emit both complete candidates");

    std::filesystem::remove(adjacentOnnxManifest(settings.pose_onnx));
    settings.compute_mode = ComputeMode::Cpu;
    requireThrows([&] { buildLaunchPlan(settings, false); },
        "CPU accepted a missing adjacent pose manifest");
    tree.makeFile(L"pose.onnx.manifest.json");
    settings.ppe_labels = L"   ";
    requireThrows([&] { buildLaunchPlan(settings, false); },
        "CPU accepted missing PPE labels");
}

void testPpeOnlyOmitsPoseArguments() {
    TemporaryTree tree;
    auto settings = baseSettings(tree);
    settings.analytics_mode = AnalyticsMode::PpeOnly;
    settings.ppe_engine = tree.makeFile(L"ppe-only.engine");
    settings.compute_mode = ComputeMode::Cuda;
    const auto cuda = buildLaunchPlan(settings, false);
    require(cuda.has_cuda_candidate && contains(cuda.arguments, L"ppe-only")
            && contains(cuda.arguments, L"--ppe-engine")
            && !contains(cuda.arguments, L"--pose-engine"),
        "PPE-only CUDA plan required or emitted a pose engine");

    settings.ppe_onnx = tree.makeFile(L"ppe-only.onnx");
    tree.makeFile(L"ppe-only.onnx.manifest.json");
    settings.compute_mode = ComputeMode::Cpu;
    const auto cpu = buildLaunchPlan(settings, false);
    require(cpu.has_cpu_candidate && contains(cpu.arguments, L"--ppe-onnx")
            && !contains(cpu.arguments, L"--pose-onnx")
            && !contains(cpu.arguments, L"--ppe-engine"),
        "PPE-only CPU plan required or emitted pose, or emitted CUDA models");

    settings.compute_mode = ComputeMode::Auto;
    const auto automatic = buildLaunchPlan(settings, false);
    require(automatic.has_cuda_candidate && automatic.has_cpu_candidate
            && contains(automatic.arguments, L"--ppe-engine")
            && contains(automatic.arguments, L"--ppe-onnx")
            && !contains(automatic.arguments, L"--pose-engine")
            && !contains(automatic.arguments, L"--pose-onnx"),
        "PPE-only Auto plan did not preserve both candidates or leaked pose arguments");
}

void testWindowsQuoting() {
    require(quoteWindowsArgument(L"plain") == L"plain", "Plain argument changed");
    require(quoteWindowsArgument(L"") == L"\"\"", "Empty argument was not quoted");
    require(quoteWindowsArgument(L"two words") == L"\"two words\"",
        "Whitespace argument was not quoted");
    require(quoteWindowsArgument(L"say\"hello") == L"\"say\\\"hello\"",
        "Embedded quote was not escaped");
    require(quoteWindowsArgument(L"C:\\path with space\\")
            == L"\"C:\\path with space\\\\\"",
        "Trailing backslash was not doubled before the closing quote");
    require(buildWindowsCommandLine({L"C:\\Program Files\\Cuajone\\cuajone_native.exe",
                                     L"--source", L"clip \"A\".mp4"})
            == L"\"C:\\Program Files\\Cuajone\\cuajone_native.exe\" --source \"clip \\\"A\\\".mp4\"",
        "Command line assembly is not CreateProcess-compatible");
}

void testCredentialRedaction() {
    require(redactRtspCredentials(
                "Opening rtsp://user:secret@camera.example:554/live and "
                "rtsps://operator:p%40ss@[2001:db8::1]/stream")
            == "Opening rtsp://***@camera.example:554/live and "
               "rtsps://***@[2001:db8::1]/stream",
        "RTSP credentials were persisted in launcher log text");
    require(redactRtspCredentials("rtsp://camera.example/live")
            == "rtsp://camera.example/live",
        "Credential-free RTSP URL changed");
}

void testSavedCameraProfileNames() {
    require(isValidSavedCameraProfileName(L"CAM_CUAJONE_01"),
        "Camera ID format was rejected");
    require(isValidSavedCameraProfileName(L"Gate 2-East"),
        "Safe saved camera profile name was rejected");
    for (const std::wstring_view invalid : {
             L"", L"camera:554", L"user@camera", L"camera/path", L"camera?query"}) {
        require(!isValidSavedCameraProfileName(invalid),
            "Saved camera profile name accepted a reserved character");
    }
    require(savedCameraCredentialTarget(L"CAM_CUAJONE_01")
                == L"NexoAI Vision/RTSP/CAM_CUAJONE_01",
        "Saved camera credential target changed");
    requireThrows([] { savedCameraCredentialTarget(L"camera@host"); },
        "Saved camera credential target accepted an unsafe profile name");
}

}  // namespace

int main() {
    const std::vector<std::pair<std::string, std::function<void()>>> tests{
        {"RTSP camera source requirement", testRtspCameraSourceIsRequired},
        {"launcher model matrix and arguments", testModelMatrixAndArguments},
        {"PPE-only pose omission", testPpeOnlyOmitsPoseArguments},
        {"Windows command-line quoting", testWindowsQuoting},
        {"RTSP credential redaction", testCredentialRedaction},
        {"saved camera profile names", testSavedCameraProfileNames},
    };
    for (const auto& [name, test] : tests) {
        try {
            test();
            std::cout << "PASS: " << name << '\n';
        } catch (const std::exception& error) {
            std::cerr << "FAIL: " << name << ": " << error.what() << '\n';
            return 1;
        }
    }
    std::cout << tests.size() << " launcher tests passed\n";
    return 0;
}
