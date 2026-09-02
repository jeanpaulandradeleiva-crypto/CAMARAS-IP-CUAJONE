// SPDX-License-Identifier: AGPL-3.0-only

#include "cuajone/launcher_support.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
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
    settings.managed_model_root = tree.root();
    return settings;
}

void addTensorRtModels(LauncherSettings& settings, const TemporaryTree& tree) {
    static_cast<void>(settings);
    tree.makeFile(L"ppe.engine");
    tree.makeFile(L"pose.engine");
}

void addOnnxModels(LauncherSettings& settings, const TemporaryTree& tree) {
    static_cast<void>(settings);
    tree.makeFile(L"ppe.onnx");
    tree.makeFile(L"ppe.onnx.manifest.json");
    tree.makeFile(L"pose.onnx");
    tree.makeFile(L"pose.onnx.manifest.json");
}

void testCameraOrVideoSourceIsRequired() {
    TemporaryTree tree;
    auto settings = baseSettings(tree);
    addTensorRtModels(settings, tree);
    settings.source = tree.makeFile(L"source with space.mp4").wstring();
    const auto video_plan = buildLaunchPlan(settings, false);
    const auto source = std::ranges::find(video_plan.arguments, L"--source");
    require(source != video_plan.arguments.end() && std::next(source) != video_plan.arguments.end()
            && *std::next(source) == settings.source,
        "Launcher did not emit the local video file as the --source argument");
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

    tree.makeFile(L"ppe.plan");
    tree.makeFile(L"pose.plan");
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

    std::filesystem::remove(tree.root() / L"ppe.engine");
    std::filesystem::remove(tree.root() / L"pose.engine");
    settings.compute_mode = ComputeMode::Cuda;
    const auto cuda_onnx = buildLaunchPlan(settings, false);
    require(cuda_onnx.has_cuda_candidate && contains(cuda_onnx.arguments, L"--ppe-onnx")
            && contains(cuda_onnx.arguments, L"--pose-onnx")
            && !contains(cuda_onnx.arguments, L"--ppe-engine"),
        "CUDA plan rejected the validated ONNX model family");

    tree.makeFile(L"ppe.pb");
    tree.makeFile(L"ppe.pb.manifest.json");
    std::filesystem::remove(tree.root() / L"ppe.onnx");
    requireThrows([&] { buildLaunchPlan(settings, false); },
        "CPU accepted a model with a non-ONNX extension");
    tree.makeFile(L"ppe.onnx");

    addTensorRtModels(settings, tree);
    settings.compute_mode = ComputeMode::Auto;
    const auto both = buildLaunchPlan(settings, false);
    require(both.has_cuda_candidate && both.has_cpu_candidate
            && contains(both.arguments, L"--ppe-engine")
            && contains(both.arguments, L"--ppe-onnx"),
        "Auto did not emit both complete candidates");

    std::filesystem::remove(tree.root() / L"pose.onnx.manifest.json");
    settings.compute_mode = ComputeMode::Cpu;
    requireThrows([&] { buildLaunchPlan(settings, false); },
        "CPU accepted a missing adjacent pose manifest");
    tree.makeFile(L"pose.onnx.manifest.json");
}

void testPpeOnlyOmitsPoseArguments() {
    TemporaryTree tree;
    auto settings = baseSettings(tree);
    settings.analytics_mode = AnalyticsMode::PpeOnly;
    tree.makeFile(L"ppe.engine");
    settings.compute_mode = ComputeMode::Cuda;
    const auto cuda = buildLaunchPlan(settings, false);
    require(cuda.has_cuda_candidate && contains(cuda.arguments, L"ppe-only")
            && contains(cuda.arguments, L"--ppe-engine")
            && !contains(cuda.arguments, L"--pose-engine"),
        "PPE-only CUDA plan required or emitted a pose engine");

    tree.makeFile(L"ppe.onnx");
    tree.makeFile(L"ppe.onnx.manifest.json");
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

void testOperationalSettingsAndArguments() {
    TemporaryTree tree;
    auto settings = baseSettings(tree);
    addOnnxModels(settings, tree);
    settings.compute_mode = ComputeMode::Cpu;
    settings.image_size = 960;
    settings.ppe_class_confidences = {
        0.01F, 0.12F, 0.23F, 0.34F, 0.45F, 0.56F, 0.67F, 0.78F,
    };
    const auto plan = buildLaunchPlan(settings, false);
    require(contains(plan.arguments, L"--imgsz") && contains(plan.arguments, L"960"),
        "Launcher did not emit the selected imgsz");
    require(std::count(plan.arguments.begin(), plan.arguments.end(), L"--ppe-class-conf") == 8,
        "Launcher did not emit exactly eight PPE class thresholds");
    require(std::count(plan.arguments.begin(), plan.arguments.end(), L"--ppe-enabled") == 7,
        "Launcher did not emit exactly seven PPE switches");
    require(contains(plan.arguments, L"Gloves=0.01")
            && contains(plan.arguments, L"lentes_protectores=0.78"),
        "Launcher class threshold order or formatting changed");
    require(contains(plan.arguments, L"--show"),
        "Launcher did not show annotated video by default");

    settings.show_window = false;
    require(!contains(buildLaunchPlan(settings, false).arguments, L"--show"),
        "Launcher emitted --show after the user disabled annotated video");
    settings.show_window = true;
    settings.ppe_enabled[0] = false;
    require(contains(buildLaunchPlan(settings, false).arguments, L"Gloves=0"),
        "Launcher did not emit a disabled PPE switch");

    settings.ppe_class_confidences[0] = 0.123F;
    requireThrows([&] { buildLaunchPlan(settings, false); },
        "Launcher accepted a PPE threshold with more-than-hundredth precision");
    settings.ppe_class_confidences[0] = 0.01F;

    settings.image_size = 800;
    requireThrows([&] { buildLaunchPlan(settings, false); },
        "Launcher accepted an unsupported imgsz");
    settings.image_size = 640;
    settings.runtime_options.emplace_back(L"--ppe-onnx", L"unmanaged.onnx");
    requireThrows([&] { buildLaunchPlan(settings, false); },
        "Launcher accepted a model-path UI option");
}

void testPpeConfidenceThresholdParsing() {
    require(parsePpeConfidenceThreshold(L"0") == 0.0F
            && parsePpeConfidenceThreshold(L"0.42") == 0.42F
            && parsePpeConfidenceThreshold(L"1") == 1.0F,
        "PPE confidence parser rejected valid inclusive values");
    require(formatPpeConfidenceThreshold(parsePpeConfidenceThreshold(L"0.4")) == L"0.40"
            && formatPpeConfidenceThreshold(parsePpeConfidenceThreshold(L"1.00")) == L"1.00",
        "PPE confidence parser did not canonicalize to hundredths");
    for (const std::wstring_view invalid : {
             L"", L".42", L"0.", L"0.001", L"1.01", L"-0.01", L"nan", L"1,0", L"1e0"}) {
        requireThrows([&] { static_cast<void>(parsePpeConfidenceThreshold(invalid)); },
            "PPE confidence parser accepted invalid manual input");
    }
    requireThrows([] { static_cast<void>(formatPpeConfidenceThreshold(0.123F)); },
        "PPE confidence formatter accepted more-than-hundredth precision");
}

void testPreferencesPersistenceAndUiContract() {
    TemporaryTree tree;
    const auto path = tree.root() / L"settings" / L"operator-settings-v1.txt";
    require(OperatorPreferences{}.show_window && LauncherSettings{}.show_window,
        "Annotated video does not default to enabled");
    OperatorPreferences preferences;
    preferences.language = UiLanguage::Spanish;
    preferences.theme = ThemeMode::Dark;
    preferences.image_size = 1280;
    preferences.ppe_class_confidences[0] = 0.11F;
    preferences.ppe_class_confidences[7] = 0.88F;
    preferences.show_window = false;
    preferences.ppe_enabled[0] = false;
    preferences.ppe_enabled[6] = false;
    saveOperatorPreferencesAtomic(path, preferences);
    const auto loaded = loadOperatorPreferences(path);
    require(loaded.language == UiLanguage::Spanish && loaded.theme == ThemeMode::Dark
            && loaded.image_size == 1280
            && loaded.ppe_class_confidences[0] == 0.11F
            && loaded.ppe_class_confidences[7] == 0.88F
            && !loaded.show_window && !loaded.ppe_enabled[0] && !loaded.ppe_enabled[6],
        "Operator preferences did not roundtrip");
    std::ofstream(path, std::ios::binary | std::ios::trunc)
        << "schema_version=1\n"
        << "language=es\n"
        << "theme=dark\n"
        << "imgsz=1280\n"
        << "ppe_class_conf=Gloves:0.11,Person:0.30,Safety_boots:0.30,Vest:0.30,"
        << "respirador:0.30,tapaorejas:0.30,Hard_hat:0.30,lentes_protectores:0.88\n";
    const auto legacy = loadOperatorPreferences(path);
    require(legacy.language == UiLanguage::Spanish && legacy.theme == ThemeMode::Dark
            && legacy.image_size == 1280 && legacy.ppe_class_confidences[0] == 0.11F
            && legacy.ppe_class_confidences[7] == 0.88F && legacy.show_window
            && std::ranges::all_of(legacy.ppe_enabled, [](bool enabled) { return enabled; }),
        "Legacy preferences did not retain settings and default annotated video to enabled");
    std::ofstream(path, std::ios::binary | std::ios::trunc) << "corrupt";
    const auto fallback = loadOperatorPreferences(path);
    require(fallback.language == UiLanguage::English && fallback.theme == ThemeMode::Light
            && fallback.image_size == 640 && fallback.ppe_class_confidences[0] == 0.30F
            && fallback.show_window,
        "Corrupt preferences did not fail closed to defaults");

    const auto controls = visibleLauncherControlKeys();
    for (const std::string_view required : {
             "imgsz", "Gloves", "Person", "Safety_boots", "Vest", "respirador",
             "tapaorejas", "Hard_hat", "lentes_protectores", "language_icon", "theme_icon"}) {
        require(std::ranges::find(controls, required) != controls.end(),
            "Required launcher control is missing from the UI contract");
    }
    for (const std::string_view forbidden : {
             "ppe_engine", "pose_engine", "ppe_onnx", "pose_onnx", "ppe_labels"}) {
        require(std::ranges::find(controls, forbidden) == controls.end(),
            "Model-path control leaked into the UI contract");
    }
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
    require(buildWindowsCommandLine({L"C:\\Program Files\\Cuajone\\NexoAIVision.exe",
                                     L"--source", L"clip \"A\".mp4"})
            == L"\"C:\\Program Files\\Cuajone\\NexoAIVision.exe\" --source \"clip \\\"A\\\".mp4\"",
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
        {"camera or video source requirement", testCameraOrVideoSourceIsRequired},
        {"launcher model matrix and arguments", testModelMatrixAndArguments},
        {"PPE-only pose omission", testPpeOnlyOmitsPoseArguments},
        {"operational settings and arguments", testOperationalSettingsAndArguments},
        {"PPE confidence threshold parsing", testPpeConfidenceThresholdParsing},
        {"preferences persistence and UI contract", testPreferencesPersistenceAndUiContract},
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
