// SPDX-License-Identifier: AGPL-3.0-only

#include "cuajone/launcher_support.hpp"

#include <windows.h>
#include <commctrl.h>
#include <shlobj.h>
#include <shobjidl.h>

#include <algorithm>
#include <atomic>
#include <cwctype>
#include <cwchar>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace cuajone::launcher;

constexpr wchar_t kWindowClass[] = L"CuajoneLauncherWindow";
constexpr wchar_t kBundledPpeLabels[] = L"Gloves,Hard_hat,Mask,Person,Safety_boots,Vest";
constexpr UINT kProcessFinished = WM_APP + 1;
constexpr ULONGLONG kGracefulStopMilliseconds = 30000;

std::filesystem::path siblingRuntime();

enum ControlId : int {
    SourceEdit = 100,
    SourceLabelEdit,
    LoadEnvButton,
    OutputEdit,
    OutputBrowse,
    AnalyticsCombo,
    ComputeCombo,
    PpeEngineEdit,
    PpeEngineBrowse,
    PoseEngineEdit,
    PoseEngineBrowse,
    PpeOnnxEdit,
    PpeOnnxBrowse,
    PoseOnnxEdit,
    PoseOnnxBrowse,
    LabelsEdit,
    ShowCheck,
    ValidateButton,
    StartButton,
    StopButton,
    StatusText,
    LogPathEdit,
};

struct LauncherWindow {
    HWND window{};
    HWND source{};
    HWND source_label{};
    HWND output{};
    HWND analytics{};
    HWND compute{};
    HWND ppe_engine{};
    HWND pose_engine{};
    HWND ppe_onnx{};
    HWND pose_onnx{};
    HWND labels{};
    HWND show{};
    HWND validate{};
    HWND start{};
    HWND stop{};
    HWND status{};
    HWND log_path{};
    std::vector<std::pair<std::wstring, std::wstring>> runtime_options;
    HFONT font{};
    std::filesystem::path program_data;
    HANDLE process{};
    HANDLE job{};
    DWORD process_id{};
    std::thread waiter;
    std::thread output_pump;
    std::atomic_bool stop_requested{};
    std::atomic<ULONGLONG> stop_deadline{};
    bool close_requested{};
};

std::wstring editText(HWND control) {
    const int length = GetWindowTextLengthW(control);
    std::wstring result(static_cast<std::size_t>(length) + 1, L'\0');
    GetWindowTextW(control, result.data(), length + 1);
    result.resize(static_cast<std::size_t>(length));
    return result;
}

void setText(HWND control, const std::filesystem::path& value) {
    SetWindowTextW(control, value.c_str());
}

void setStatus(LauncherWindow& state, std::wstring_view text) {
    SetWindowTextW(state.status, std::wstring(text).c_str());
}

std::wstring trim(std::wstring value) {
    const auto first = std::find_if_not(value.begin(), value.end(), [](wchar_t character) {
        return std::iswspace(character) != 0;
    });
    const auto last = std::find_if_not(value.rbegin(), value.rend(), [](wchar_t character) {
        return std::iswspace(character) != 0;
    }).base();
    return first >= last ? std::wstring{} : std::wstring(first, last);
}

std::wstring upper(std::wstring value) {
    std::transform(value.begin(), value.end(), value.begin(), [](wchar_t character) {
        return static_cast<wchar_t>(std::towupper(character));
    });
    return value;
}

std::map<std::wstring, std::wstring> readEnvFile(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Could not open the selected .env file");
    std::map<std::wstring, std::wstring> values;
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.starts_with("\xEF\xBB\xBF")) line.erase(0, 3);
        const int length = MultiByteToWideChar(
            CP_UTF8, MB_ERR_INVALID_CHARS, line.data(), static_cast<int>(line.size()), nullptr, 0);
        if (length <= 0) throw std::runtime_error("The selected .env file is not valid UTF-8");
        std::wstring wide(static_cast<std::size_t>(length), L'\0');
        MultiByteToWideChar(
            CP_UTF8, MB_ERR_INVALID_CHARS, line.data(), static_cast<int>(line.size()), wide.data(), length);
        wide = trim(wide);
        if (wide.empty() || wide.starts_with(L"#")) continue;
        const std::size_t separator = wide.find(L'=');
        if (separator == std::wstring::npos) continue;
        std::wstring key = upper(trim(wide.substr(0, separator)));
        std::wstring value = trim(wide.substr(separator + 1));
        if (value.size() >= 2 && value.front() == L'"' && value.back() == L'"') {
            value = value.substr(1, value.size() - 2);
        }
        if (!key.empty()) values.insert_or_assign(std::move(key), std::move(value));
    }
    return values;
}

std::optional<std::wstring> envValue(
    const std::map<std::wstring, std::wstring>& values,
    std::wstring_view key) {
    const auto found = values.find(std::wstring(key));
    if (found == values.end() || found->second.empty()) return std::nullopt;
    return found->second;
}

std::filesystem::path resolveEnvPath(
    const std::filesystem::path& env_path,
    const std::wstring& configured) {
    std::filesystem::path path(configured);
    return path.is_absolute() ? path : env_path.parent_path() / path;
}

HWND createControl(
    LauncherWindow& state,
    DWORD extended_style,
    const wchar_t* class_name,
    const wchar_t* text,
    DWORD style,
    int x,
    int y,
    int width,
    int height,
    int id) {
    HWND control = CreateWindowExW(
        extended_style, class_name, text, WS_CHILD | WS_VISIBLE | style,
        x, y, width, height, state.window,
        reinterpret_cast<HMENU>(static_cast<INT_PTR>(id)),
        GetModuleHandleW(nullptr), nullptr);
    if (control == nullptr) throw std::runtime_error("Could not create launcher control");
    SendMessageW(control, WM_SETFONT, reinterpret_cast<WPARAM>(state.font), TRUE);
    return control;
}

void createLabel(LauncherWindow& state, const wchar_t* text, int x, int y, int width) {
    createControl(state, 0, L"STATIC", text, SS_LEFT, x, y + 4, width, 22, 0);
}

HWND createEdit(LauncherWindow& state, int id, int x, int y, int width, bool read_only = false) {
    DWORD style = WS_TABSTOP | ES_AUTOHSCROLL;
    if (read_only) style |= ES_READONLY;
    HWND control = createControl(
        state, WS_EX_CLIENTEDGE, L"EDIT", L"", style, x, y, width, 25, id);
    SendMessageW(control, EM_SETLIMITTEXT, 32767, 0);
    return control;
}

HWND createBrowseButton(LauncherWindow& state, int id, int y) {
    return createControl(
        state, 0, L"BUTTON", L"Browse...", WS_TABSTOP | BS_PUSHBUTTON,
        778, y, 92, 25, id);
}

std::filesystem::path knownProgramData() {
    PWSTR raw_path = nullptr;
    const HRESULT result = SHGetKnownFolderPath(FOLDERID_ProgramData, KF_FLAG_DEFAULT, nullptr, &raw_path);
    if (FAILED(result) || raw_path == nullptr) {
        throw std::runtime_error("SHGetKnownFolderPath(FOLDERID_ProgramData) failed");
    }
    std::filesystem::path path(raw_path);
    CoTaskMemFree(raw_path);
    return path / L"Cuajone PPE Monitor" / L"runtime";
}

bool hasModelBundleAt(const std::filesystem::path& root) {
    std::error_code error;
    const auto models = root / L"models";
    return std::filesystem::is_regular_file(models / L"ppe.engine", error)
        || std::filesystem::is_regular_file(models / L"pose.engine", error)
        || std::filesystem::is_regular_file(models / L"ppe.onnx", error)
        || std::filesystem::is_regular_file(models / L"pose.onnx", error);
}

std::filesystem::path preferredModelRoot() {
    const std::filesystem::path install_root = siblingRuntime().parent_path();
    const std::filesystem::path install_models = install_root / L"models";
    const std::filesystem::path program_data_models = knownProgramData() / L"models";
    if (hasModelBundleAt(install_root)) {
        return install_models;
    }
    if (hasModelBundleAt(program_data_models.parent_path())) {
        return program_data_models;
    }
    return install_models;
}

void createControls(LauncherWindow& state) {
    state.font = static_cast<HFONT>(GetStockObject(DEFAULT_GUI_FONT));
    constexpr int label_x = 16;
    constexpr int edit_x = 146;
    constexpr int edit_width = 620;

    createLabel(state, L"RTSP camera", label_x, 18, 120);
    state.source = createEdit(state, SourceEdit, edit_x, 18, 724);
    SetWindowTextW(state.source, L"rtsp://");

    createLabel(state, L"Camera ID", label_x, 54, 120);
    state.source_label = createEdit(state, SourceLabelEdit, edit_x, 54, edit_width);
    createControl(
        state, 0, L"BUTTON", L"Load .env...", WS_TABSTOP | BS_PUSHBUTTON,
        778, 54, 92, 25, LoadEnvButton);

    createLabel(state, L"Output folder", label_x, 90, 120);
    state.output = createEdit(state, OutputEdit, edit_x, 90, edit_width);
    createBrowseButton(state, OutputBrowse, 90);

    createLabel(state, L"Analytics mode", label_x, 128, 120);
    state.analytics = createControl(
        state, 0, WC_COMBOBOXW, L"", WS_TABSTOP | CBS_DROPDOWNLIST,
        edit_x, 126, 220, 200, AnalyticsCombo);
    SendMessageW(state.analytics, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(L"PPE only"));
    SendMessageW(state.analytics, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(L"PPE + fall"));
    SendMessageW(state.analytics, CB_SETCURSEL, 1, 0);

    createLabel(state, L"Compute", 410, 128, 75);
    state.compute = createControl(
        state, 0, WC_COMBOBOXW, L"", WS_TABSTOP | CBS_DROPDOWNLIST,
        486, 126, 180, 200, ComputeCombo);
    SendMessageW(state.compute, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(L"Auto"));
    SendMessageW(state.compute, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(L"CUDA"));
    SendMessageW(state.compute, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(L"CPU"));
    SendMessageW(state.compute, CB_SETCURSEL, 0, 0);

    createLabel(state, L"PPE engine", label_x, 164, 120);
    state.ppe_engine = createEdit(state, PpeEngineEdit, edit_x, 164, edit_width);
    createBrowseButton(state, PpeEngineBrowse, 164);

    createLabel(state, L"Pose engine", label_x, 200, 120);
    state.pose_engine = createEdit(state, PoseEngineEdit, edit_x, 200, edit_width);
    createBrowseButton(state, PoseEngineBrowse, 200);

    createLabel(state, L"PPE ONNX", label_x, 236, 120);
    state.ppe_onnx = createEdit(state, PpeOnnxEdit, edit_x, 236, edit_width);
    createBrowseButton(state, PpeOnnxBrowse, 236);

    createLabel(state, L"Pose ONNX", label_x, 272, 120);
    state.pose_onnx = createEdit(state, PoseOnnxEdit, edit_x, 272, edit_width);
    createBrowseButton(state, PoseOnnxBrowse, 272);

    createLabel(state, L"PPE labels", label_x, 308, 120);
    state.labels = createEdit(state, LabelsEdit, edit_x, 308, edit_width);

    state.show = createControl(
        state, 0, L"BUTTON", L"Show annotated video window",
        WS_TABSTOP | BS_AUTOCHECKBOX, edit_x, 344, 260, 24, ShowCheck);

    state.validate = createControl(
        state, 0, L"BUTTON", L"Validate", WS_TABSTOP | BS_PUSHBUTTON,
        edit_x, 384, 110, 30, ValidateButton);
    state.start = createControl(
        state, 0, L"BUTTON", L"Start", WS_TABSTOP | BS_DEFPUSHBUTTON,
        270, 384, 110, 30, StartButton);
    state.stop = createControl(
        state, 0, L"BUTTON", L"Stop", WS_TABSTOP | BS_PUSHBUTTON,
        394, 384, 110, 30, StopButton);
    EnableWindow(state.stop, FALSE);

    createLabel(state, L"Status", label_x, 436, 120);
    state.status = createControl(
        state, WS_EX_CLIENTEDGE, L"STATIC", L"Ready", SS_LEFT | SS_CENTERIMAGE,
        edit_x, 432, 724, 30, StatusText);
    createLabel(state, L"Log path", label_x, 480, 120);
    state.log_path = createEdit(state, LogPathEdit, edit_x, 476, 724, true);

    state.program_data = knownProgramData();
    const auto models = preferredModelRoot();
    setText(state.output, state.program_data / L"output");
    setText(state.ppe_engine, models / L"ppe.engine");
    setText(state.pose_engine, models / L"pose.engine");
    setText(state.ppe_onnx, models / L"ppe.onnx");
    setText(state.pose_onnx, models / L"pose.onnx");
    SetWindowTextW(state.labels, kBundledPpeLabels);
    SetWindowTextW(state.source_label, L"CAM_CUAJONE_01");
    setText(state.log_path, state.program_data / L"logs");

    std::error_code models_error;
    const auto install_models = siblingRuntime().parent_path() / L"models";
    const auto program_data_models = state.program_data / L"models";
    const bool any_model_installed = std::filesystem::is_regular_file(install_models / L"ppe.engine", models_error)
        || std::filesystem::is_regular_file(install_models / L"pose.engine", models_error)
        || std::filesystem::is_regular_file(install_models / L"ppe.onnx", models_error)
        || std::filesystem::is_regular_file(install_models / L"pose.onnx", models_error)
        || std::filesystem::is_regular_file(program_data_models / L"ppe.engine", models_error)
        || std::filesystem::is_regular_file(program_data_models / L"pose.engine", models_error)
        || std::filesystem::is_regular_file(program_data_models / L"ppe.onnx", models_error)
        || std::filesystem::is_regular_file(program_data_models / L"pose.onnx", models_error);
    if (!any_model_installed) {
        setStatus(
            state,
            L"No AI models are installed yet. Browse to ppe.engine / pose.engine or ppe.onnx / pose.onnx under the app folder or ProgramData.");
    }
}

std::filesystem::path pickFile(HWND owner, const COMDLG_FILTERSPEC* filters, UINT filter_count) {
    IFileOpenDialog* dialog = nullptr;
    HRESULT result = CoCreateInstance(
        CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER,
        IID_PPV_ARGS(&dialog));
    if (FAILED(result)) return {};
    DWORD options{};
    dialog->GetOptions(&options);
    dialog->SetOptions(options | FOS_FORCEFILESYSTEM | FOS_FILEMUSTEXIST | FOS_PATHMUSTEXIST);
    dialog->SetFileTypes(filter_count, filters);
    result = dialog->Show(owner);
    std::filesystem::path path;
    if (SUCCEEDED(result)) {
        IShellItem* item = nullptr;
        if (SUCCEEDED(dialog->GetResult(&item))) {
            PWSTR raw_path = nullptr;
            if (SUCCEEDED(item->GetDisplayName(SIGDN_FILESYSPATH, &raw_path))) {
                path = raw_path;
                CoTaskMemFree(raw_path);
            }
            item->Release();
        }
    }
    dialog->Release();
    return path;
}

void selectCombo(HWND control, std::wstring_view value, int fallback) {
    const LRESULT index = SendMessageW(
        control, CB_FINDSTRINGEXACT, static_cast<WPARAM>(-1),
        reinterpret_cast<LPARAM>(std::wstring(value).c_str()));
    SendMessageW(control, CB_SETCURSEL, index == CB_ERR ? fallback : index, 0);
}

std::wstring lowerExtension(const std::filesystem::path& path) {
    std::wstring extension = path.extension().wstring();
    std::transform(extension.begin(), extension.end(), extension.begin(), [](wchar_t character) {
        return static_cast<wchar_t>(std::towlower(character));
    });
    return extension;
}

void loadEnv(LauncherWindow& state) {
    static constexpr COMDLG_FILTERSPEC filters[] = {
        {L"Environment file", L"*.env;*.txt"}, {L"All files", L"*.*"},
    };
    const std::filesystem::path env_path = pickFile(
        state.window, filters, static_cast<UINT>(std::size(filters)));
    if (env_path.empty()) return;

    const auto values = readEnvFile(env_path);
    const auto applyText = [&](std::wstring_view key, HWND control) {
        if (const auto value = envValue(values, key)) SetWindowTextW(control, value->c_str());
    };
    applyText(L"RTSP_URL", state.source);
    applyText(L"CAMERA_ID", state.source_label);
    if (const auto output = envValue(values, L"OUTPUT_DIR")) {
        setText(state.output, resolveEnvPath(env_path, *output));
    }
    if (const auto mode = envValue(values, L"ANALYTICS_MODE")) {
        selectCombo(state.analytics, *mode == L"ppe-only" ? L"PPE only" : L"PPE + fall", 1);
    }
    if (const auto show = envValue(values, L"SHOW_WINDOW")) {
        SendMessageW(state.show, BM_SETCHECK,
            (*show == L"1" || upper(*show) == L"TRUE") ? BST_CHECKED : BST_UNCHECKED, 0);
    }

    std::vector<std::wstring> ignored;
    const auto applyModel = [&](std::wstring_view key, HWND onnx, HWND engine) {
        const auto configured = envValue(values, key);
        if (!configured) return;
        const std::filesystem::path model = resolveEnvPath(env_path, *configured);
        if (lowerExtension(model) == L".onnx") SetWindowTextW(onnx, model.c_str());
        else if (lowerExtension(model) == L".engine") SetWindowTextW(engine, model.c_str());
        else ignored.emplace_back(std::wstring(key) + L" (native runtime requires .onnx or .engine)");
    };
    applyModel(L"PPE_MODEL_PATH", state.ppe_onnx, state.ppe_engine);
    applyModel(L"POSE_MODEL_PATH", state.pose_onnx, state.pose_engine);

    state.runtime_options.clear();
    const auto append = [&](std::wstring_view key, std::wstring_view option) {
        if (const auto value = envValue(values, key)) {
            state.runtime_options.emplace_back(std::wstring(option), *value);
        }
    };
    append(L"TARGET_INFERENCE_FPS", L"--target-fps");
    append(L"POSE_CONF", L"--pose-conf");
    append(L"PPE_CONF", L"--ppe-conf");
    append(L"IOU_THRESHOLD", L"--nms-iou");
    append(L"EPP_WINDOW", L"--ppe-window");
    append(L"EPP_MIN_SAMPLES", L"--ppe-min-samples");
    append(L"EPP_PRESENT_RATIO", L"--ppe-present-ratio");
    append(L"EPP_ALERT_COOLDOWN_S", L"--ppe-cooldown");
    append(L"FALL_CONFIRM_FRAMES", L"--fall-confirm-frames");
    append(L"FALL_RESET_FRAMES", L"--fall-reset-frames");
    append(L"FALL_ALERT_COOLDOWN_S", L"--fall-cooldown");
    append(L"FALL_ASPECT_RATIO", L"--fall-aspect-ratio");
    append(L"FALL_TORSO_ANGLE_DEG", L"--fall-torso-angle");
    append(L"FALL_DESCENT_RATIO", L"--fall-descent-ratio");
    append(L"FALL_NEAR_FLOOR_RATIO", L"--fall-near-floor-ratio");
    if (const auto ttl = envValue(values, L"TRACK_TTL_S")) {
        state.runtime_options.emplace_back(L"--ppe-track-ttl", *ttl);
        state.runtime_options.emplace_back(L"--fall-track-ttl", *ttl);
    }
    append(L"RECONNECT_DELAY_S", L"--reconnect-delay");
    append(L"RTSP_TRANSPORT", L"--rtsp-transport");
    append(L"RTSP_OPEN_TIMEOUT_MS", L"--capture-open-timeout-ms");
    append(L"RTSP_READ_TIMEOUT_MS", L"--capture-read-timeout-ms");
    if (const auto device = envValue(values, L"YOLO_DEVICE")) {
        if (upper(*device) != L"CPU" && upper(*device) != L"AUTO") {
            state.runtime_options.emplace_back(L"--device", *device);
        }
    }

    for (const std::wstring_view key : {
             L"YOLO_TRACKER", L"USE_FP16", L"SHOW_TEMPORARY_TRACK_ID",
             L"EXCEL_EXPORT_EVERY_EVENTS", L"POSE_IMGSZ", L"PPE_IMGSZ",
             L"RTSP_SOCKET_TIMEOUT_S"}) {
        if (envValue(values, key)) ignored.emplace_back(key);
    }
    if (ignored.empty()) {
        setStatus(state, L"Loaded .env settings");
    } else {
        setStatus(state, L"Loaded .env; Python-only settings were ignored");
    }
}

std::filesystem::path pickFolder(HWND owner) {
    IFileOpenDialog* dialog = nullptr;
    HRESULT result = CoCreateInstance(
        CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER,
        IID_PPV_ARGS(&dialog));
    if (FAILED(result)) return {};
    DWORD options{};
    dialog->GetOptions(&options);
    dialog->SetOptions(options | FOS_FORCEFILESYSTEM | FOS_PICKFOLDERS | FOS_PATHMUSTEXIST);
    result = dialog->Show(owner);
    std::filesystem::path path;
    if (SUCCEEDED(result)) {
        IShellItem* item = nullptr;
        if (SUCCEEDED(dialog->GetResult(&item))) {
            PWSTR raw_path = nullptr;
            if (SUCCEEDED(item->GetDisplayName(SIGDN_FILESYSPATH, &raw_path))) {
                path = raw_path;
                CoTaskMemFree(raw_path);
            }
            item->Release();
        }
    }
    dialog->Release();
    return path;
}

LauncherSettings readSettings(const LauncherWindow& state) {
    LauncherSettings settings;
    settings.source = editText(state.source);
    settings.output = editText(state.output);
    settings.analytics_mode = SendMessageW(state.analytics, CB_GETCURSEL, 0, 0) == 0
        ? AnalyticsMode::PpeOnly : AnalyticsMode::PpeFall;
    const LRESULT compute = SendMessageW(state.compute, CB_GETCURSEL, 0, 0);
    settings.compute_mode = compute == 1
        ? ComputeMode::Cuda : (compute == 2 ? ComputeMode::Cpu : ComputeMode::Auto);
    settings.ppe_engine = editText(state.ppe_engine);
    settings.pose_engine = editText(state.pose_engine);
    settings.ppe_onnx = editText(state.ppe_onnx);
    settings.pose_onnx = editText(state.pose_onnx);
    settings.ppe_labels = editText(state.labels);
    settings.source_label = editText(state.source_label);
    settings.runtime_options = state.runtime_options;
    settings.show_window = SendMessageW(state.show, BM_GETCHECK, 0, 0) == BST_CHECKED;
    return settings;
}

std::filesystem::path siblingRuntime() {
    std::wstring module_path(32768, L'\0');
    const DWORD length = GetModuleFileNameW(
        nullptr, module_path.data(), static_cast<DWORD>(module_path.size()));
    if (length == 0 || length == static_cast<DWORD>(module_path.size())) {
        throw std::runtime_error("Could not resolve the launcher executable path");
    }
    module_path.resize(length);
    return std::filesystem::path(module_path).parent_path() / L"cuajone_native.exe";
}

std::filesystem::path nextLogPath(const LauncherWindow& state) {
    SYSTEMTIME now{};
    GetLocalTime(&now);
    wchar_t name[64]{};
    swprintf_s(
        name, L"cuajone-%04u%02u%02u-%02u%02u%02u-%03u.log",
        now.wYear, now.wMonth, now.wDay,
        now.wHour, now.wMinute, now.wSecond, now.wMilliseconds);
    return state.program_data / L"logs" / name;
}

void setRunning(LauncherWindow& state, bool running) {
    EnableWindow(state.validate, !running);
    EnableWindow(state.start, !running);
    EnableWindow(state.stop, running);
}

void closeProcessHandles(LauncherWindow& state) {
    if (state.process != nullptr) CloseHandle(state.process);
    if (state.job != nullptr) CloseHandle(state.job);
    state.process = nullptr;
    state.job = nullptr;
    state.process_id = 0;
}

void writeLog(HANDLE log, std::string_view text) {
    std::size_t offset = 0;
    while (offset < text.size()) {
        const DWORD remaining = static_cast<DWORD>(std::min<std::size_t>(
            text.size() - offset, MAXDWORD));
        DWORD written{};
        if (!WriteFile(log, text.data() + offset, remaining, &written, nullptr)
            || written == 0) {
            return;
        }
        offset += written;
    }
}

void launchRuntime(LauncherWindow& state, bool preflight) {
    if (state.process != nullptr) return;
    const LauncherSettings settings = readSettings(state);
    const LaunchPlan plan = buildLaunchPlan(settings, preflight);
    const std::filesystem::path runtime = siblingRuntime();
    std::error_code file_error;
    if (!std::filesystem::is_regular_file(runtime, file_error) || file_error) {
        throw std::runtime_error("Sibling cuajone_native.exe was not found");
    }

    std::filesystem::create_directories(settings.output);
    const std::filesystem::path log_path = nextLogPath(state);
    std::filesystem::create_directories(log_path.parent_path());
    setText(state.log_path, log_path);

    SECURITY_ATTRIBUTES security{sizeof(security), nullptr, TRUE};
    HANDLE pipe_read = nullptr;
    HANDLE pipe_write = nullptr;
    if (!CreatePipe(&pipe_read, &pipe_write, &security, 0)
        || !SetHandleInformation(pipe_read, HANDLE_FLAG_INHERIT, 0)) {
        if (pipe_read != nullptr) CloseHandle(pipe_read);
        if (pipe_write != nullptr) CloseHandle(pipe_write);
        throw std::runtime_error("Could not create the child output pipe");
    }
    HANDLE log = CreateFileW(
        log_path.c_str(), GENERIC_WRITE,
        FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
        CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (log == INVALID_HANDLE_VALUE) {
        CloseHandle(pipe_read);
        CloseHandle(pipe_write);
        throw std::runtime_error("Could not create the ProgramData log");
    }
    HANDLE null_input = CreateFileW(
        L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
        &security, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (null_input == INVALID_HANDLE_VALUE) {
        CloseHandle(pipe_read);
        CloseHandle(pipe_write);
        CloseHandle(log);
        throw std::runtime_error("Could not open NUL for child input");
    }

    HANDLE job = CreateJobObjectW(nullptr, nullptr);
    if (job == nullptr) {
        CloseHandle(pipe_read);
        CloseHandle(pipe_write);
        CloseHandle(null_input);
        CloseHandle(log);
        throw std::runtime_error("Could not create the runtime Job Object");
    }
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    if (!SetInformationJobObject(
            job, JobObjectExtendedLimitInformation, &limits, sizeof(limits))) {
        CloseHandle(job);
        CloseHandle(pipe_read);
        CloseHandle(pipe_write);
        CloseHandle(null_input);
        CloseHandle(log);
        throw std::runtime_error("Could not configure KILL_ON_JOB_CLOSE");
    }

    std::vector<std::wstring> command_arguments{runtime.wstring()};
    command_arguments.insert(
        command_arguments.end(), plan.arguments.begin(), plan.arguments.end());
    std::wstring command_line = buildWindowsCommandLine(command_arguments);
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdInput = null_input;
    startup.hStdOutput = pipe_write;
    startup.hStdError = pipe_write;
    PROCESS_INFORMATION process{};
    const std::wstring working_directory = runtime.parent_path().wstring();
    const BOOL created = CreateProcessW(
        runtime.c_str(), command_line.data(), nullptr, nullptr, TRUE,
        CREATE_SUSPENDED | CREATE_NEW_PROCESS_GROUP,
        nullptr, working_directory.c_str(), &startup, &process);
    CloseHandle(null_input);
    CloseHandle(pipe_write);
    if (!created) {
        CloseHandle(pipe_read);
        CloseHandle(log);
        CloseHandle(job);
        throw std::runtime_error("CreateProcessW failed for cuajone_native.exe");
    }
    if (!AssignProcessToJobObject(job, process.hProcess)) {
        TerminateProcess(process.hProcess, 1);
        CloseHandle(process.hThread);
        CloseHandle(process.hProcess);
        CloseHandle(pipe_read);
        CloseHandle(log);
        CloseHandle(job);
        throw std::runtime_error("Could not assign cuajone_native.exe to its Job Object");
    }
    if (ResumeThread(process.hThread) == static_cast<DWORD>(-1)) {
        TerminateJobObject(job, 1);
        CloseHandle(process.hThread);
        CloseHandle(process.hProcess);
        CloseHandle(pipe_read);
        CloseHandle(log);
        CloseHandle(job);
        throw std::runtime_error("Could not resume cuajone_native.exe");
    }
    CloseHandle(process.hThread);

    state.process = process.hProcess;
    state.job = job;
    state.process_id = process.dwProcessId;
    state.stop_requested.store(false, std::memory_order_relaxed);
    state.stop_deadline.store(0, std::memory_order_relaxed);

    try {
        state.output_pump = std::thread([pipe_read, log] {
            std::string pending;
            char buffer[4096];
            DWORD bytes_read{};
            while (ReadFile(pipe_read, buffer, sizeof(buffer), &bytes_read, nullptr) && bytes_read != 0) {
                pending.append(buffer, bytes_read);
                std::size_t newline{};
                while ((newline = pending.find('\n')) != std::string::npos) {
                    const std::string safe = redactRtspCredentials(
                        std::string_view(pending).substr(0, newline + 1));
                    writeLog(log, safe);
                    pending.erase(0, newline + 1);
                }
            }
            if (!pending.empty()) writeLog(log, redactRtspCredentials(pending));
            FlushFileBuffers(log);
            CloseHandle(log);
            CloseHandle(pipe_read);
        });

        state.waiter = std::thread([&state, preflight, process_handle = state.process, job_handle = state.job] {
            bool forced = false;
            while (WaitForSingleObject(process_handle, 100) == WAIT_TIMEOUT) {
                if (!forced && state.stop_requested.load(std::memory_order_relaxed)
                    && GetTickCount64() >= state.stop_deadline.load(std::memory_order_relaxed)) {
                    TerminateJobObject(job_handle, 130);
                    forced = true;
                }
            }
            DWORD exit_code = 1;
            GetExitCodeProcess(process_handle, &exit_code);
            PostMessageW(
                state.window, kProcessFinished,
                static_cast<WPARAM>(exit_code), static_cast<LPARAM>(preflight));
        });
    } catch (...) {
        TerminateJobObject(state.job, 1);
        if (state.waiter.joinable()) state.waiter.join();
        if (state.output_pump.joinable()) state.output_pump.join();
        else {
            CloseHandle(log);
            CloseHandle(pipe_read);
        }
        closeProcessHandles(state);
        throw;
    }
    setRunning(state, true);
    setStatus(state, preflight ? L"Validating configuration..." : L"Runtime is running");
}

void requestStop(LauncherWindow& state) {
    if (state.process == nullptr || state.stop_requested.exchange(true)) return;
    state.stop_deadline.store(
        GetTickCount64() + kGracefulStopMilliseconds, std::memory_order_relaxed);
    const BOOL signaled = GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT, state.process_id);
    EnableWindow(state.stop, FALSE);
    setStatus(
        state,
        signaled
            ? L"Stopping gracefully; forced termination follows after 30 seconds"
            : L"CTRL_BREAK delivery failed; forced termination follows after 30 seconds");
}

void finishProcess(LauncherWindow& state, DWORD exit_code, bool preflight) {
    const bool stopped = state.stop_requested.load(std::memory_order_relaxed);
    if (state.waiter.joinable()) state.waiter.join();
    if (state.output_pump.joinable()) state.output_pump.join();
    closeProcessHandles(state);
    setRunning(state, false);
    if (stopped) {
        setStatus(state, L"Runtime stopped (exit code " + std::to_wstring(exit_code) + L")");
    } else if (exit_code == 0) {
        setStatus(state, preflight ? L"Validation passed" : L"Runtime completed successfully");
    } else {
        setStatus(
            state,
            (preflight ? L"Validation failed (exit code " : L"Runtime failed (exit code ")
                + std::to_wstring(exit_code) + L"); see log");
    }
    state.stop_requested.store(false, std::memory_order_relaxed);
    if (state.close_requested) DestroyWindow(state.window);
}

void showError(LauncherWindow& state, const std::exception& error) {
    const std::string narrow(error.what());
    const std::wstring message(narrow.begin(), narrow.end());
    setStatus(state, L"Configuration error");
    MessageBoxW(state.window, message.c_str(), L"Cuajone launcher", MB_OK | MB_ICONERROR);
}

void browseInto(LauncherWindow& state, int id) {
    static constexpr COMDLG_FILTERSPEC engine_filters[] = {
        {L"TensorRT engine", L"*.engine"}, {L"All files", L"*.*"},
    };
    static constexpr COMDLG_FILTERSPEC onnx_filters[] = {
        {L"ONNX model", L"*.onnx"}, {L"All files", L"*.*"},
    };
    if (id == OutputBrowse) {
        const auto path = pickFolder(state.window);
        if (!path.empty()) setText(state.output, path);
        return;
    }
    HWND destination = nullptr;
    const COMDLG_FILTERSPEC* filters = engine_filters;
    UINT filter_count = static_cast<UINT>(std::size(engine_filters));
    if (id == PpeEngineBrowse || id == PoseEngineBrowse) {
        destination = id == PpeEngineBrowse ? state.ppe_engine : state.pose_engine;
    } else if (id == PpeOnnxBrowse || id == PoseOnnxBrowse) {
        destination = id == PpeOnnxBrowse ? state.ppe_onnx : state.pose_onnx;
        filters = onnx_filters;
        filter_count = static_cast<UINT>(std::size(onnx_filters));
    }
    if (destination != nullptr) {
        const auto path = pickFile(state.window, filters, filter_count);
        if (!path.empty()) setText(destination, path);
    }
}

void shutdown(LauncherWindow& state) {
    if (state.job != nullptr) TerminateJobObject(state.job, 130);
    if (state.waiter.joinable()) state.waiter.join();
    if (state.output_pump.joinable()) state.output_pump.join();
    closeProcessHandles(state);
}

LRESULT CALLBACK windowProcedure(HWND window, UINT message, WPARAM wparam, LPARAM lparam) {
    auto* state = reinterpret_cast<LauncherWindow*>(GetWindowLongPtrW(window, GWLP_USERDATA));
    if (message == WM_NCCREATE) {
        const auto* create = reinterpret_cast<CREATESTRUCTW*>(lparam);
        state = static_cast<LauncherWindow*>(create->lpCreateParams);
        state->window = window;
        SetWindowLongPtrW(window, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(state));
    }
    if (state == nullptr) return DefWindowProcW(window, message, wparam, lparam);

    try {
        switch (message) {
            case WM_CREATE:
                createControls(*state);
                return 0;
            case WM_COMMAND: {
                const int id = LOWORD(wparam);
                if (id == ValidateButton) launchRuntime(*state, true);
                else if (id == StartButton) launchRuntime(*state, false);
                else if (id == StopButton) requestStop(*state);
                else if (id == LoadEnvButton) loadEnv(*state);
                else if (id == OutputBrowse
                    || id == PpeEngineBrowse || id == PoseEngineBrowse
                    || id == PpeOnnxBrowse || id == PoseOnnxBrowse) {
                    browseInto(*state, id);
                }
                return 0;
            }
            case kProcessFinished:
                finishProcess(*state, static_cast<DWORD>(wparam), lparam != 0);
                return 0;
            case WM_CLOSE:
                if (state->process != nullptr) {
                    state->close_requested = true;
                    requestStop(*state);
                } else {
                    DestroyWindow(window);
                }
                return 0;
            case WM_DESTROY:
                shutdown(*state);
                PostQuitMessage(0);
                return 0;
            default:
                break;
        }
    } catch (const std::exception& error) {
        showError(*state, error);
        return message == WM_CREATE ? -1 : 0;
    }
    return DefWindowProcW(window, message, wparam, lparam);
}

}  // namespace

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE, PWSTR, int show_command) {
    const HRESULT com = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
    if (FAILED(com)) {
        MessageBoxW(nullptr, L"COM initialization failed", L"Cuajone launcher", MB_OK | MB_ICONERROR);
        return 1;
    }
    if (AllocConsole()) {
        const HWND console = GetConsoleWindow();
        if (console != nullptr) ShowWindow(console, SW_HIDE);
        SetConsoleCtrlHandler(nullptr, TRUE);
    }
    INITCOMMONCONTROLSEX common_controls{sizeof(common_controls), ICC_STANDARD_CLASSES};
    InitCommonControlsEx(&common_controls);

    WNDCLASSEXW window_class{};
    window_class.cbSize = sizeof(window_class);
    window_class.lpfnWndProc = windowProcedure;
    window_class.hInstance = instance;
    window_class.hCursor = LoadCursorW(nullptr, IDC_ARROW);
    window_class.hIcon = LoadIconW(nullptr, IDI_APPLICATION);
    window_class.hbrBackground = reinterpret_cast<HBRUSH>(COLOR_WINDOW + 1);
    window_class.lpszClassName = kWindowClass;
    if (RegisterClassExW(&window_class) == 0) {
        MessageBoxW(nullptr, L"Window registration failed", L"Cuajone launcher", MB_OK | MB_ICONERROR);
        FreeConsole();
        CoUninitialize();
        return 1;
    }

    LauncherWindow state;
    HWND window = CreateWindowExW(
        0, kWindowClass, L"Cuajone PPE Monitor",
        WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX,
        CW_USEDEFAULT, CW_USEDEFAULT, 906, 570,
        nullptr, nullptr, instance, &state);
    if (window == nullptr) {
        MessageBoxW(nullptr, L"Launcher window creation failed", L"Cuajone launcher", MB_OK | MB_ICONERROR);
        FreeConsole();
        CoUninitialize();
        return 1;
    }
    ShowWindow(window, show_command);
    UpdateWindow(window);

    MSG message{};
    while (GetMessageW(&message, nullptr, 0, 0) > 0) {
        if (!IsDialogMessageW(window, &message)) {
            TranslateMessage(&message);
            DispatchMessageW(&message);
        }
    }
    FreeConsole();
    CoUninitialize();
    return static_cast<int>(message.wParam);
}
