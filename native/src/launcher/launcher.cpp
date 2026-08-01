// SPDX-License-Identifier: AGPL-3.0-only

#include "cuajone/launcher_support.hpp"

#include <windows.h>
#include <commctrl.h>
#include <shellapi.h>
#include <shlobj.h>
#include <shobjidl.h>
#include <wincred.h>

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
#include <string_view>
#include <thread>
#include <vector>

namespace {

using namespace cuajone::launcher;

constexpr wchar_t kWindowClass[] = L"NexoAIVisionLauncherWindow";
constexpr wchar_t kProductName[] = L"NexoAI Vision";
constexpr wchar_t kLegacyProductName[] = L"Cuajone PPE Monitor";
constexpr wchar_t kBundledPpeLabels[] = L"Gloves,Hard_hat,Mask,Person,Safety_boots,Vest";
constexpr UINT kProcessFinished = WM_APP + 1;
constexpr ULONGLONG kGracefulStopMilliseconds = 30000;
constexpr COLORREF kWindowColor = RGB(246, 248, 252);
constexpr COLORREF kInputColor = RGB(255, 255, 255);
constexpr COLORREF kTextColor = RGB(31, 41, 55);
constexpr COLORREF kMutedTextColor = RGB(75, 85, 99);
constexpr COLORREF kPrimaryColor = RGB(22, 91, 170);
constexpr COLORREF kPrimaryPressedColor = RGB(17, 72, 136);

std::filesystem::path siblingRuntime();

enum ControlId : int {
    SourceEdit = 100,
    SourceLabelEdit,
    LanguageCombo,
    LoadEnvButton,
    OpenLogButton,
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
    SavedCameraCombo,
    SaveCameraButton,
    LoadCameraButton,
    DeleteCameraButton,
};

struct LocalizedText {
    HWND control{};
    const wchar_t* english{};
    const wchar_t* spanish{};
};

struct LauncherWindow {
    HWND window{};
    HWND source{};
    HWND source_label{};
    HWND saved_camera{};
    HWND save_camera{};
    HWND load_camera{};
    HWND delete_camera{};
    HWND language{};
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
    std::vector<LocalizedText> localized_text;
    std::vector<std::pair<std::wstring, std::wstring>> runtime_options;
    HFONT font{};
    HFONT heading_font{};
    HFONT button_font{};
    HBRUSH window_brush{};
    HBRUSH input_brush{};
    HBRUSH status_brush{};
    std::filesystem::path program_data;
    HANDLE process{};
    HANDLE job{};
    DWORD process_id{};
    std::thread waiter;
    std::thread output_pump;
    std::atomic_bool stop_requested{};
    std::atomic<ULONGLONG> stop_deadline{};
    bool close_requested{};
    bool spanish{};
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

void addLocalizedText(
    LauncherWindow& state,
    HWND control,
    const wchar_t* english,
    const wchar_t* spanish) {
    state.localized_text.push_back({control, english, spanish});
}

void updateAnalyticsOptions(LauncherWindow& state) {
    const LRESULT selection = SendMessageW(state.analytics, CB_GETCURSEL, 0, 0);
    SendMessageW(state.analytics, CB_RESETCONTENT, 0, 0);
    SendMessageW(
        state.analytics, CB_ADDSTRING, 0,
        reinterpret_cast<LPARAM>(state.spanish ? L"Solo EPP" : L"PPE only"));
    SendMessageW(
        state.analytics, CB_ADDSTRING, 0,
        reinterpret_cast<LPARAM>(state.spanish ? L"EPP + caídas" : L"PPE + fall"));
    SendMessageW(state.analytics, CB_SETCURSEL, selection == CB_ERR ? 1 : selection, 0);
}

void refreshLanguage(LauncherWindow& state) {
    for (const auto& text : state.localized_text) {
        SetWindowTextW(text.control, state.spanish ? text.spanish : text.english);
    }
    updateAnalyticsOptions(state);
    const std::wstring current_status = editText(state.status);
    if (current_status == L"Ready" || current_status == L"Listo") {
        setStatus(state, state.spanish ? L"Listo" : L"Ready");
    }
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

std::string utf8FromWide(std::wstring_view value) {
    if (value.empty()) return {};
    const int required = WideCharToMultiByte(
        CP_UTF8, 0, value.data(), static_cast<int>(value.size()), nullptr, 0, nullptr, nullptr);
    if (required <= 0) return {};
    std::string result(static_cast<std::size_t>(required), '\0');
    WideCharToMultiByte(
        CP_UTF8, 0, value.data(), static_cast<int>(value.size()), result.data(), required, nullptr, nullptr);
    return result;
}

std::runtime_error savedCameraProfileError(std::string_view action, std::wstring_view profile) {
    return std::runtime_error(std::string(action) + ": " + utf8FromWide(profile));
}

std::optional<std::wstring> selectedSavedCameraProfile(const LauncherWindow& state) {
    const LRESULT selection = SendMessageW(state.saved_camera, CB_GETCURSEL, 0, 0);
    if (selection == CB_ERR) return std::nullopt;
    const LRESULT length = SendMessageW(state.saved_camera, CB_GETLBTEXTLEN, selection, 0);
    if (length == CB_ERR) return std::nullopt;
    std::wstring profile(static_cast<std::size_t>(length) + 1, L'\0');
    SendMessageW(state.saved_camera, CB_GETLBTEXT, selection, reinterpret_cast<LPARAM>(profile.data()));
    profile.resize(static_cast<std::size_t>(length));
    return isValidSavedCameraProfileName(profile) ? std::optional<std::wstring>(std::move(profile)) : std::nullopt;
}

void refreshSavedCameraProfiles(LauncherWindow& state, std::wstring_view preferred = {}) {
    std::vector<std::wstring> profiles;
    PCREDENTIALW* credentials = nullptr;
    DWORD count{};
    const std::wstring filter = std::wstring(savedCameraCredentialTargetPrefix()) + L"*";
    if (CredEnumerateW(filter.c_str(), 0, &count, &credentials)) {
        for (DWORD index = 0; index < count; ++index) {
            const std::wstring_view target(credentials[index]->TargetName);
            const std::wstring_view prefix = savedCameraCredentialTargetPrefix();
            if (!target.starts_with(prefix)) continue;
            const std::wstring_view profile = target.substr(prefix.size());
            if (isValidSavedCameraProfileName(profile)) profiles.emplace_back(profile);
        }
        CredFree(credentials);
    } else if (GetLastError() != ERROR_NOT_FOUND) {
        throw std::runtime_error("Could not list saved camera profiles");
    }
    std::sort(profiles.begin(), profiles.end());
    profiles.erase(std::unique(profiles.begin(), profiles.end()), profiles.end());
    SendMessageW(state.saved_camera, CB_RESETCONTENT, 0, 0);
    for (const auto& profile : profiles) {
        const LRESULT index = SendMessageW(
            state.saved_camera, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(profile.c_str()));
        if (index != CB_ERR && profile == preferred) {
            SendMessageW(state.saved_camera, CB_SETCURSEL, index, 0);
        }
    }
}

void saveSavedCameraProfile(LauncherWindow& state) {
    const std::wstring profile = trim(editText(state.source_label));
    if (!isValidSavedCameraProfileName(profile)) {
        throw std::invalid_argument("Saved camera profile name is invalid");
    }
    const std::wstring source = trim(editText(state.source));
    validateRtspCameraUrl(source);
    if (source.size() > CRED_MAX_CREDENTIAL_BLOB_SIZE / sizeof(wchar_t)) {
        throw std::invalid_argument("RTSP camera URL is too long to save");
    }
    const std::wstring target = savedCameraCredentialTarget(profile);
    CREDENTIALW credential{};
    credential.Type = CRED_TYPE_GENERIC;
    credential.TargetName = const_cast<LPWSTR>(target.c_str());
    credential.CredentialBlobSize = static_cast<DWORD>(source.size() * sizeof(wchar_t));
    credential.CredentialBlob = reinterpret_cast<LPBYTE>(const_cast<wchar_t*>(source.data()));
    credential.Persist = CRED_PERSIST_LOCAL_MACHINE;
    if (!CredWriteW(&credential, 0)) {
        throw savedCameraProfileError("Could not save camera profile", profile);
    }
    SetWindowTextW(state.source_label, profile.c_str());
    refreshSavedCameraProfiles(state, profile);
    setStatus(state, (state.spanish ? L"Perfil de cámara guardado: " : L"Saved camera profile: ") + profile);
}

void loadSavedCameraProfile(LauncherWindow& state) {
    const auto profile = selectedSavedCameraProfile(state);
    if (!profile) throw std::invalid_argument("Select a saved camera profile");
    const std::wstring target = savedCameraCredentialTarget(*profile);
    PCREDENTIALW credential = nullptr;
    if (!CredReadW(target.c_str(), CRED_TYPE_GENERIC, 0, &credential)) {
        throw savedCameraProfileError("Could not load camera profile", *profile);
    }
    const bool valid_blob = credential->CredentialBlob != nullptr
        && credential->CredentialBlobSize != 0
        && credential->CredentialBlobSize % sizeof(wchar_t) == 0;
    std::wstring source;
    if (valid_blob) {
        const auto* text = reinterpret_cast<const wchar_t*>(credential->CredentialBlob);
        source.assign(text, credential->CredentialBlobSize / sizeof(wchar_t));
    }
    CredFree(credential);
    if (!valid_blob) {
        throw savedCameraProfileError("Saved camera profile is invalid", *profile);
    }
    try {
        validateRtspCameraUrl(source);
    } catch (const std::exception&) {
        throw savedCameraProfileError("Saved camera profile is invalid", *profile);
    }
    SetWindowTextW(state.source_label, profile->c_str());
    SetWindowTextW(state.source, source.c_str());
    setStatus(state, (state.spanish ? L"Perfil de cámara cargado: " : L"Saved camera profile loaded: ") + *profile);
}

void deleteSavedCameraProfile(LauncherWindow& state) {
    const auto profile = selectedSavedCameraProfile(state);
    if (!profile) throw std::invalid_argument("Select a saved camera profile");
    const std::wstring target = savedCameraCredentialTarget(*profile);
    if (!CredDeleteW(target.c_str(), CRED_TYPE_GENERIC, 0)) {
        throw savedCameraProfileError("Could not delete camera profile", *profile);
    }
    refreshSavedCameraProfiles(state);
    setStatus(state, (state.spanish ? L"Perfil de cámara eliminado: " : L"Saved camera profile deleted: ") + *profile);
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
    const HFONT font = std::wcscmp(class_name, L"BUTTON") == 0 && state.button_font != nullptr
        ? state.button_font
        : state.font;
    SendMessageW(control, WM_SETFONT, reinterpret_cast<WPARAM>(font), TRUE);
    return control;
}

HWND createLabel(LauncherWindow& state, const wchar_t* text, int x, int y, int width) {
    return createControl(state, 0, L"STATIC", text, SS_LEFT, x, y + 4, width, 22, 0);
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
        state, 0, L"BUTTON", L"Browse...", WS_TABSTOP | BS_OWNERDRAW,
        778, y, 92, 25, id);
}

void drawButton(const DRAWITEMSTRUCT& item) {
    const bool disabled = (item.itemState & ODS_DISABLED) != 0;
    const bool pressed = (item.itemState & (ODS_SELECTED | ODS_HOTLIGHT)) != 0;
    const int id = static_cast<int>(item.CtlID);
    const bool primary = id == StartButton;
    const COLORREF fill = disabled ? RGB(229, 231, 235)
        : primary ? (pressed ? kPrimaryPressedColor : kPrimaryColor)
        : (pressed ? RGB(219, 234, 254) : RGB(255, 255, 255));
    const COLORREF border = disabled ? RGB(209, 213, 219) : primary ? fill : RGB(203, 213, 225);
    HBRUSH brush = CreateSolidBrush(fill);
    HPEN pen = CreatePen(PS_SOLID, 1, border);
    HGDIOBJ old_brush = SelectObject(item.hDC, brush);
    HGDIOBJ old_pen = SelectObject(item.hDC, pen);
    RoundRect(item.hDC, item.rcItem.left, item.rcItem.top, item.rcItem.right, item.rcItem.bottom, 8, 8);
    SelectObject(item.hDC, old_brush);
    SelectObject(item.hDC, old_pen);
    DeleteObject(pen);
    DeleteObject(brush);

    wchar_t text[128]{};
    GetWindowTextW(item.hwndItem, text, static_cast<int>(std::size(text)));
    SetBkMode(item.hDC, TRANSPARENT);
    SetTextColor(item.hDC, disabled ? RGB(156, 163, 175) : primary ? RGB(255, 255, 255) : kTextColor);
    DrawTextW(item.hDC, text, -1, const_cast<RECT*>(&item.rcItem), DT_CENTER | DT_VCENTER | DT_SINGLELINE);
}

std::filesystem::path knownProgramData() {
    PWSTR raw_path = nullptr;
    const HRESULT result = SHGetKnownFolderPath(FOLDERID_ProgramData, KF_FLAG_DEFAULT, nullptr, &raw_path);
    if (FAILED(result) || raw_path == nullptr) {
        throw std::runtime_error("SHGetKnownFolderPath(FOLDERID_ProgramData) failed");
    }
    std::filesystem::path path(raw_path);
    CoTaskMemFree(raw_path);
    return path / kProductName / L"runtime";
}

std::filesystem::path legacyProgramData() {
    PWSTR raw_path = nullptr;
    const HRESULT result = SHGetKnownFolderPath(FOLDERID_ProgramData, KF_FLAG_DEFAULT, nullptr, &raw_path);
    if (FAILED(result) || raw_path == nullptr) {
        throw std::runtime_error("SHGetKnownFolderPath(FOLDERID_ProgramData) failed");
    }
    std::filesystem::path path(raw_path);
    CoTaskMemFree(raw_path);
    return path / kLegacyProductName / L"runtime";
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
    const std::filesystem::path legacy_program_data_models = legacyProgramData() / L"models";
    if (hasModelBundleAt(install_root)) {
        return install_models;
    }
    if (hasModelBundleAt(program_data_models.parent_path())) {
        return program_data_models;
    }
    if (hasModelBundleAt(legacy_program_data_models.parent_path())) {
        return legacy_program_data_models;
    }
    return install_models;
}

void createControls(LauncherWindow& state) {
    state.font = CreateFontW(-15, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
        OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, DEFAULT_PITCH, L"Segoe UI");
    state.heading_font = CreateFontW(-24, 0, 0, 0, FW_SEMIBOLD, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
        OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, DEFAULT_PITCH, L"Segoe UI");
    state.button_font = CreateFontW(-15, 0, 0, 0, FW_SEMIBOLD, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
        OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, DEFAULT_PITCH, L"Segoe UI");
    state.window_brush = CreateSolidBrush(kWindowColor);
    state.input_brush = CreateSolidBrush(kInputColor);
    state.status_brush = CreateSolidBrush(RGB(232, 240, 254));
    constexpr int label_x = 16;
    constexpr int edit_x = 146;
    constexpr int edit_width = 620;
    constexpr int row_y = 80;

    const HWND heading = createControl(state, 0, L"STATIC", kProductName, SS_LEFT, 16, 14, 400, 30, 0);
    SendMessageW(heading, WM_SETFONT, reinterpret_cast<WPARAM>(state.heading_font), TRUE);
    const HWND subtitle = createControl(
        state, 0, L"STATIC", L"Camera analytics control center", SS_LEFT, 18, 46, 400, 20, 0);
    addLocalizedText(state, subtitle, L"Camera analytics control center", L"Centro de control de analítica de cámaras");

    addLocalizedText(
        state, createLabel(state, L"RTSP camera", label_x, row_y, 120),
        L"RTSP camera", L"Cámara RTSP");
    state.source = createEdit(state, SourceEdit, edit_x, row_y, 724);
    SetWindowTextW(state.source, L"rtsp://");

    addLocalizedText(
        state, createLabel(state, L"Camera ID", label_x, row_y + 36, 120),
        L"Camera ID", L"ID de cámara");
    state.source_label = createEdit(state, SourceLabelEdit, edit_x, row_y + 36, 430);
    addLocalizedText(
        state, createLabel(state, L"Language", 590, row_y + 36, 72),
        L"Language", L"Idioma");
    state.language = createControl(
        state, 0, WC_COMBOBOXW, L"", WS_TABSTOP | CBS_DROPDOWNLIST,
        664, row_y + 36, 104, 200, LanguageCombo);
    SendMessageW(state.language, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(L"English"));
    SendMessageW(state.language, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(L"Español"));
    SendMessageW(state.language, CB_SETCURSEL, 0, 0);
    const HWND load_env = createControl(
        state, 0, L"BUTTON", L"Load .env...", WS_TABSTOP | BS_PUSHBUTTON,
        778, row_y + 36, 92, 25, LoadEnvButton);
    addLocalizedText(state, load_env, L"Load .env...", L"Cargar .env...");

    addLocalizedText(
        state, createLabel(state, L"Saved camera", label_x, row_y + 72, 120),
        L"Saved camera", L"Cámara guardada");
    state.saved_camera = createControl(
        state, 0, WC_COMBOBOXW, L"", WS_TABSTOP | CBS_DROPDOWNLIST,
        edit_x, row_y + 72, 430, 200, SavedCameraCombo);
    state.save_camera = createControl(
        state, 0, L"BUTTON", L"Save", WS_TABSTOP | BS_PUSHBUTTON,
        586, row_y + 72, 80, 25, SaveCameraButton);
    addLocalizedText(state, state.save_camera, L"Save", L"Guardar");
    state.load_camera = createControl(
        state, 0, L"BUTTON", L"Load", WS_TABSTOP | BS_PUSHBUTTON,
        674, row_y + 72, 80, 25, LoadCameraButton);
    addLocalizedText(state, state.load_camera, L"Load", L"Cargar");
    state.delete_camera = createControl(
        state, 0, L"BUTTON", L"Delete", WS_TABSTOP | BS_PUSHBUTTON,
        762, row_y + 72, 108, 25, DeleteCameraButton);
    addLocalizedText(state, state.delete_camera, L"Delete", L"Eliminar");

    addLocalizedText(
        state, createLabel(state, L"Output folder", label_x, row_y + 108, 120),
        L"Output folder", L"Carpeta de salida");
    state.output = createEdit(state, OutputEdit, edit_x, row_y + 108, edit_width);
    addLocalizedText(state, createBrowseButton(state, OutputBrowse, row_y + 108), L"Browse...", L"Explorar...");

    addLocalizedText(
        state, createLabel(state, L"Analytics mode", label_x, row_y + 146, 120),
        L"Analytics mode", L"Modo de análisis");
    state.analytics = createControl(
        state, 0, WC_COMBOBOXW, L"", WS_TABSTOP | CBS_DROPDOWNLIST,
        edit_x, row_y + 144, 220, 200, AnalyticsCombo);
    updateAnalyticsOptions(state);

    addLocalizedText(
        state, createLabel(state, L"Compute", 410, row_y + 146, 75),
        L"Compute", L"Cómputo");
    state.compute = createControl(
        state, 0, WC_COMBOBOXW, L"", WS_TABSTOP | CBS_DROPDOWNLIST,
        486, row_y + 144, 180, 200, ComputeCombo);
    SendMessageW(state.compute, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(L"Auto"));
    SendMessageW(state.compute, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(L"CUDA"));
    SendMessageW(state.compute, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(L"CPU"));
    SendMessageW(state.compute, CB_SETCURSEL, 0, 0);

    addLocalizedText(
        state, createLabel(state, L"PPE engine", label_x, row_y + 182, 120),
        L"PPE engine", L"Motor EPP");
    state.ppe_engine = createEdit(state, PpeEngineEdit, edit_x, row_y + 182, edit_width);
    addLocalizedText(state, createBrowseButton(state, PpeEngineBrowse, row_y + 182), L"Browse...", L"Explorar...");

    addLocalizedText(
        state, createLabel(state, L"Pose engine", label_x, row_y + 218, 120),
        L"Pose engine", L"Motor de pose");
    state.pose_engine = createEdit(state, PoseEngineEdit, edit_x, row_y + 218, edit_width);
    addLocalizedText(state, createBrowseButton(state, PoseEngineBrowse, row_y + 218), L"Browse...", L"Explorar...");

    addLocalizedText(
        state, createLabel(state, L"PPE ONNX", label_x, row_y + 254, 120),
        L"PPE ONNX", L"ONNX EPP");
    state.ppe_onnx = createEdit(state, PpeOnnxEdit, edit_x, row_y + 254, edit_width);
    addLocalizedText(state, createBrowseButton(state, PpeOnnxBrowse, row_y + 254), L"Browse...", L"Explorar...");

    addLocalizedText(
        state, createLabel(state, L"Pose ONNX", label_x, row_y + 290, 120),
        L"Pose ONNX", L"ONNX de pose");
    state.pose_onnx = createEdit(state, PoseOnnxEdit, edit_x, row_y + 290, edit_width);
    addLocalizedText(state, createBrowseButton(state, PoseOnnxBrowse, row_y + 290), L"Browse...", L"Explorar...");

    addLocalizedText(
        state, createLabel(state, L"PPE labels", label_x, row_y + 326, 120),
        L"PPE labels", L"Etiquetas EPP");
    state.labels = createEdit(state, LabelsEdit, edit_x, row_y + 326, edit_width);

    state.show = createControl(
        state, 0, L"BUTTON", L"Show annotated video window",
        WS_TABSTOP | BS_AUTOCHECKBOX, edit_x, row_y + 362, 340, 24, ShowCheck);
    addLocalizedText(
        state, state.show, L"Show annotated video window", L"Mostrar ventana de video anotada");

    state.validate = createControl(
        state, 0, L"BUTTON", L"Validate", WS_TABSTOP | BS_OWNERDRAW,
        edit_x, row_y + 402, 110, 32, ValidateButton);
    addLocalizedText(state, state.validate, L"Validate", L"Validar");
    state.start = createControl(
        state, 0, L"BUTTON", L"Start", WS_TABSTOP | BS_OWNERDRAW,
        270, row_y + 402, 110, 32, StartButton);
    addLocalizedText(state, state.start, L"Start", L"Iniciar");
    state.stop = createControl(
        state, 0, L"BUTTON", L"Stop", WS_TABSTOP | BS_OWNERDRAW,
        394, row_y + 402, 110, 32, StopButton);
    addLocalizedText(state, state.stop, L"Stop", L"Detener");
    EnableWindow(state.stop, FALSE);

    addLocalizedText(
        state, createLabel(state, L"Status", label_x, row_y + 454, 120),
        L"Status", L"Estado");
    state.status = createControl(
        state, WS_EX_CLIENTEDGE, L"STATIC", L"Ready", SS_LEFT | SS_CENTERIMAGE,
        edit_x, row_y + 450, 724, 32, StatusText);
    addLocalizedText(
        state, createLabel(state, L"Log path", label_x, row_y + 498, 120),
        L"Log path", L"Ruta del log");
    state.log_path = createEdit(state, LogPathEdit, edit_x, row_y + 494, 620, true);
    const HWND open_log = createControl(
        state, 0, L"BUTTON", L"Open log", WS_TABSTOP | BS_PUSHBUTTON,
        778, row_y + 494, 92, 25, OpenLogButton);
    addLocalizedText(state, open_log, L"Open log", L"Abrir log");

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
    refreshSavedCameraProfiles(state);

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
    refreshLanguage(state);
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
        SendMessageW(state.analytics, CB_SETCURSEL, *mode == L"ppe-only" ? 0 : 1, 0);
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
    MessageBoxW(state.window, message.c_str(), kProductName, MB_OK | MB_ICONERROR);
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
            case WM_ERASEBKGND: {
                RECT client{};
                GetClientRect(window, &client);
                FillRect(reinterpret_cast<HDC>(wparam), &client, state->window_brush);
                return 1;
            }
            case WM_CTLCOLOREDIT:
            case WM_CTLCOLORLISTBOX:
                SetTextColor(reinterpret_cast<HDC>(wparam), kTextColor);
                SetBkColor(reinterpret_cast<HDC>(wparam), kInputColor);
                return reinterpret_cast<LRESULT>(state->input_brush);
            case WM_CTLCOLORSTATIC:
                if (reinterpret_cast<HWND>(lparam) == state->status) {
                    SetTextColor(reinterpret_cast<HDC>(wparam), kPrimaryColor);
                    SetBkColor(reinterpret_cast<HDC>(wparam), RGB(232, 240, 254));
                    return reinterpret_cast<LRESULT>(state->status_brush);
                }
                SetTextColor(reinterpret_cast<HDC>(wparam), kMutedTextColor);
                SetBkMode(reinterpret_cast<HDC>(wparam), TRANSPARENT);
                return reinterpret_cast<LRESULT>(state->window_brush);
            case WM_CTLCOLORBTN:
                SetTextColor(reinterpret_cast<HDC>(wparam), kTextColor);
                SetBkMode(reinterpret_cast<HDC>(wparam), TRANSPARENT);
                return reinterpret_cast<LRESULT>(state->window_brush);
            case WM_DRAWITEM:
                if (const auto* item = reinterpret_cast<const DRAWITEMSTRUCT*>(lparam);
                    item->CtlType == ODT_BUTTON) {
                    drawButton(*item);
                    return TRUE;
                }
                break;
            case WM_COMMAND: {
                const int id = LOWORD(wparam);
                if (id == ValidateButton) launchRuntime(*state, true);
                else if (id == OpenLogButton) {
                    const std::wstring path = editText(state->log_path);
                    if (!std::filesystem::is_regular_file(path)) {
                        throw std::runtime_error("The current log file does not exist yet");
                    }
                    if (reinterpret_cast<INT_PTR>(ShellExecuteW(
                            state->window, L"open", path.c_str(), nullptr, nullptr, SW_SHOWNORMAL)) <= 32) {
                        throw std::runtime_error("Could not open the current log file");
                    }
                }
                else if (id == StartButton) launchRuntime(*state, false);
                else if (id == StopButton) requestStop(*state);
                else if (id == SaveCameraButton) saveSavedCameraProfile(*state);
                else if (id == LoadCameraButton) loadSavedCameraProfile(*state);
                else if (id == DeleteCameraButton) deleteSavedCameraProfile(*state);
                else if (id == LanguageCombo && HIWORD(wparam) == CBN_SELCHANGE) {
                    state->spanish = SendMessageW(state->language, CB_GETCURSEL, 0, 0) == 1;
                    refreshLanguage(*state);
                }
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
                DeleteObject(state->status_brush);
                DeleteObject(state->input_brush);
                DeleteObject(state->window_brush);
                DeleteObject(state->button_font);
                DeleteObject(state->heading_font);
                DeleteObject(state->font);
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
        MessageBoxW(nullptr, L"COM initialization failed", kProductName, MB_OK | MB_ICONERROR);
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
    window_class.hIcon = LoadIconW(instance, MAKEINTRESOURCEW(101));
    window_class.hIconSm = window_class.hIcon;
    window_class.hbrBackground = nullptr;
    window_class.lpszClassName = kWindowClass;
    if (RegisterClassExW(&window_class) == 0) {
        MessageBoxW(nullptr, L"Window registration failed", kProductName, MB_OK | MB_ICONERROR);
        FreeConsole();
        CoUninitialize();
        return 1;
    }

    LauncherWindow state;
    HWND window = CreateWindowExW(
        0, kWindowClass, kProductName,
        WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX,
        CW_USEDEFAULT, CW_USEDEFAULT, 906, 650,
        nullptr, nullptr, instance, &state);
    if (window == nullptr) {
        MessageBoxW(nullptr, L"Launcher window creation failed", kProductName, MB_OK | MB_ICONERROR);
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
