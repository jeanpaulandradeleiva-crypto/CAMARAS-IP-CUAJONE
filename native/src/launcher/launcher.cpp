// SPDX-License-Identifier: AGPL-3.0-only

#include "cuajone/launcher_support.hpp"

#include <windows.h>
#include <commctrl.h>
#include <dwmapi.h>
#include <shellapi.h>
#include <shlobj.h>
#include <shobjidl.h>
#include <wincred.h>

#include <algorithm>
#include <atomic>
#include <cmath>
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
using cuajone::kAllowedImageSizes;
using cuajone::kPpeOutputLabels;

constexpr wchar_t kWindowClass[] = L"NexoAIVisionLauncherWindow";
constexpr wchar_t kProductName[] = L"NexoAI Vision";
// Runtime executable produced by the CMake target; keep in sync with CUAJONE_PRODUCT_EXE.
constexpr wchar_t kRuntimeExecutable[] = L"NexoAIVision.exe";

std::string runtimeExecutableName() {
    // The runtime executable name is ASCII-only; no locale conversion needed.
    std::string name;
    for (wchar_t ch : kRuntimeExecutable) {
        name.push_back(static_cast<char>(ch));
    }
    return name;
}
constexpr UINT kProcessFinished = WM_APP + 1;
constexpr ULONGLONG kGracefulStopMilliseconds = 30000;

struct LauncherWindow;
std::filesystem::path siblingRuntime();
void persistPreferences(LauncherWindow& state);
void showError(LauncherWindow& state, const std::exception& error);
LRESULT CALLBACK thresholdWheelProcedure(
    HWND window, UINT message, WPARAM wparam, LPARAM lparam, UINT_PTR, DWORD_PTR reference);

enum ControlId : int {
    SourceEdit = 100,
    SourceLabelEdit,
    LanguageButton,
    ThemeButton,
    LoadEnvButton,
    OpenLogButton,
    OutputEdit,
    OutputBrowse,
    SourceBrowse,
    AnalyticsCombo,
    ComputeCombo,
    ImageSizeCombo,
    PpeThresholdBase = 200,
    PpeEnabledBase = 220,
    ShowCheck = 300,
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
    HWND theme_button{};
    HWND output{};
    HWND analytics{};
    HWND compute{};
    HWND image_size{};
    std::array<HWND, kPpeOutputLabels.size()> ppe_thresholds{};
    std::array<HWND, cuajone::kPpeItemCount> ppe_enabled{};
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
    std::filesystem::path managed_model_root;
    std::filesystem::path preferences_path;
    OperatorPreferences preferences;
    HANDLE process{};
    HANDLE job{};
    DWORD process_id{};
    std::thread waiter;
    std::thread output_pump;
    std::atomic_bool stop_requested{};
    std::atomic<ULONGLONG> stop_deadline{};
    bool close_requested{};
    bool spanish{};
    bool dark{};
};

struct Palette {
    COLORREF window;
    COLORREF input;
    COLORREF text;
    COLORREF muted;
    COLORREF primary;
    COLORREF primary_pressed;
    COLORREF status;
    COLORREF border;
    COLORREF disabled;
};

Palette palette(const LauncherWindow& state) {
    if (state.dark) {
        return {
            RGB(17, 24, 39), RGB(31, 41, 55), RGB(243, 244, 246), RGB(209, 213, 219),
            RGB(59, 130, 246), RGB(37, 99, 235), RGB(30, 58, 96), RGB(75, 85, 99),
            RGB(55, 65, 81),
        };
    }
    return {
        RGB(246, 248, 252), RGB(255, 255, 255), RGB(31, 41, 55), RGB(75, 85, 99),
        RGB(22, 91, 170), RGB(17, 72, 136), RGB(232, 240, 254), RGB(203, 213, 225),
        RGB(229, 231, 235),
    };
}

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
    SetWindowTextW(
        state.language,
        state.spanish ? L"Cambiar idioma a inglés" : L"Switch language to Spanish");
    SetWindowTextW(
        state.theme_button,
        state.dark
            ? (state.spanish ? L"Cambiar a tema claro" : L"Switch to light theme")
            : (state.spanish ? L"Cambiar a tema oscuro" : L"Switch to dark theme"));
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

void drawButton(const LauncherWindow& state, const DRAWITEMSTRUCT& item) {
    const Palette colors = palette(state);
    const bool disabled = (item.itemState & ODS_DISABLED) != 0;
    const bool pressed = (item.itemState & (ODS_SELECTED | ODS_HOTLIGHT)) != 0;
    const int id = static_cast<int>(item.CtlID);
    const bool primary = id == StartButton;
    const COLORREF fill = disabled ? colors.disabled
        : primary ? (pressed ? colors.primary_pressed : colors.primary)
        : (pressed ? colors.status : colors.input);
    const COLORREF border = disabled ? colors.border : primary ? fill : colors.border;
    HBRUSH brush = CreateSolidBrush(fill);
    HPEN pen = CreatePen(PS_SOLID, 1, border);
    HGDIOBJ old_brush = SelectObject(item.hDC, brush);
    HGDIOBJ old_pen = SelectObject(item.hDC, pen);
    RoundRect(item.hDC, item.rcItem.left, item.rcItem.top, item.rcItem.right, item.rcItem.bottom, 8, 8);
    SelectObject(item.hDC, old_brush);
    SelectObject(item.hDC, old_pen);
    DeleteObject(pen);
    DeleteObject(brush);

    SetBkMode(item.hDC, TRANSPARENT);
    SetTextColor(item.hDC, disabled ? RGB(156, 163, 175) : primary ? RGB(255, 255, 255) : colors.text);
    RECT content = item.rcItem;
    if (id == LanguageButton) {
        const int center_x = (content.left + content.right) / 2;
        const int center_y = (content.top + content.bottom) / 2;
        Ellipse(item.hDC, center_x - 8, center_y - 8, center_x + 8, center_y + 8);
        MoveToEx(item.hDC, center_x - 8, center_y, nullptr);
        LineTo(item.hDC, center_x + 8, center_y);
        Arc(item.hDC, center_x - 4, center_y - 8, center_x + 4, center_y + 8,
            center_x, center_y - 8, center_x, center_y + 8);
        return;
    }
    if (id == ThemeButton) {
        const int center_x = (content.left + content.right) / 2;
        const int center_y = (content.top + content.bottom) / 2;
        if (state.dark) {
            HBRUSH moon = CreateSolidBrush(colors.text);
            HGDIOBJ old = SelectObject(item.hDC, moon);
            Ellipse(item.hDC, center_x - 7, center_y - 8, center_x + 8, center_y + 7);
            SelectObject(item.hDC, old);
            DeleteObject(moon);
            HBRUSH cutout = CreateSolidBrush(fill);
            old = SelectObject(item.hDC, cutout);
            Ellipse(item.hDC, center_x - 1, center_y - 9, center_x + 9, center_y + 1);
            SelectObject(item.hDC, old);
            DeleteObject(cutout);
        } else {
            Ellipse(item.hDC, center_x - 5, center_y - 5, center_x + 5, center_y + 5);
            for (int offset : {-10, 10}) {
                MoveToEx(item.hDC, center_x + offset, center_y, nullptr);
                LineTo(item.hDC, center_x + (offset > 0 ? 7 : -7), center_y);
                MoveToEx(item.hDC, center_x, center_y + offset, nullptr);
                LineTo(item.hDC, center_x, center_y + (offset > 0 ? 7 : -7));
            }
        }
        return;
    }
    wchar_t text[128]{};
    GetWindowTextW(item.hwndItem, text, static_cast<int>(std::size(text)));
    DrawTextW(item.hDC, text, -1, &content, DT_CENTER | DT_VCENTER | DT_SINGLELINE);
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

std::filesystem::path knownLocalAppData() {
    PWSTR raw_path = nullptr;
    const HRESULT result = SHGetKnownFolderPath(FOLDERID_LocalAppData, KF_FLAG_DEFAULT, nullptr, &raw_path);
    if (FAILED(result) || raw_path == nullptr) {
        throw std::runtime_error("SHGetKnownFolderPath(FOLDERID_LocalAppData) failed");
    }
    std::filesystem::path path(raw_path);
    CoTaskMemFree(raw_path);
    return path / kProductName / L"operator-settings-v1.txt";
}

std::filesystem::path preferredModelRoot() {
    return siblingRuntime().parent_path() / L"models";
}

void replaceBrush(HBRUSH& brush, COLORREF color) {
    if (brush != nullptr) DeleteObject(brush);
    brush = CreateSolidBrush(color);
}

void applyTheme(LauncherWindow& state) {
    const Palette colors = palette(state);
    replaceBrush(state.window_brush, colors.window);
    replaceBrush(state.input_brush, colors.input);
    replaceBrush(state.status_brush, colors.status);
    const COLORREF caption = colors.window;
    const COLORREF caption_text = colors.text;
    DwmSetWindowAttribute(
        state.window, DWMWA_CAPTION_COLOR, &caption, sizeof(caption));
    DwmSetWindowAttribute(
        state.window, DWMWA_TEXT_COLOR, &caption_text, sizeof(caption_text));
    RedrawWindow(state.window, nullptr, nullptr,
        RDW_INVALIDATE | RDW_ERASE | RDW_ALLCHILDREN | RDW_FRAME);
}

void addTooltip(LauncherWindow& state, HWND control, const wchar_t* text) {
    HWND tooltip = CreateWindowExW(
        WS_EX_TOPMOST, TOOLTIPS_CLASSW, nullptr,
        WS_POPUP | TTS_ALWAYSTIP | TTS_NOPREFIX,
        CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT,
        state.window, nullptr, GetModuleHandleW(nullptr), nullptr);
    if (tooltip == nullptr) throw std::runtime_error("Could not create launcher tooltip");
    TOOLINFOW info{sizeof(info)};
    info.uFlags = TTF_IDISHWND | TTF_SUBCLASS;
    info.hwnd = state.window;
    info.uId = reinterpret_cast<UINT_PTR>(control);
    info.lpszText = const_cast<LPWSTR>(text);
    SendMessageW(tooltip, TTM_ADDTOOLW, 0, reinterpret_cast<LPARAM>(&info));
}

HWND createClosedCombo(
    LauncherWindow& state,
    int id,
    int x,
    int y,
    int width) {
    return createControl(
        state, 0, WC_COMBOBOXW, L"", WS_TABSTOP | CBS_DROPDOWNLIST,
        x, y, width, 300, id);
}

void populateThresholdCombo(HWND combo, float selected) {
    for (int hundredths = 0; hundredths <= 100; ++hundredths) {
        wchar_t value[8]{};
        swprintf_s(value, L"%d.%02d", hundredths / 100, hundredths % 100);
        SendMessageW(combo, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(value));
    }
    SetWindowTextW(combo, formatPpeConfidenceThreshold(selected).c_str());
    SendMessageW(combo, CB_SETCURSEL,
        static_cast<WPARAM>(std::lround(selected * 100.0F)), 0);
}

void setThresholdComboValue(HWND combo, int hundredths) {
    const int clamped = std::clamp(hundredths, 0, 100);
    const std::wstring text = formatPpeConfidenceThreshold(static_cast<float>(clamped) / 100.0F);
    SendMessageW(combo, CB_SETCURSEL, clamped, 0);
    SetWindowTextW(combo, text.c_str());
}

HWND createThresholdCombo(LauncherWindow& state, int id, int x, int y, int width) {
    return createControl(
        state, 0, WC_COMBOBOXW, L"", WS_TABSTOP | CBS_DROPDOWN | CBS_AUTOHSCROLL,
        x, y, width, 300, id);
}

void createControls(LauncherWindow& state) {
    state.font = CreateFontW(-15, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
        OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, DEFAULT_PITCH, L"Segoe UI");
    state.heading_font = CreateFontW(-24, 0, 0, 0, FW_SEMIBOLD, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
        OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, DEFAULT_PITCH, L"Segoe UI");
    state.button_font = CreateFontW(-15, 0, 0, 0, FW_SEMIBOLD, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
        OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, DEFAULT_PITCH, L"Segoe UI");
    state.spanish = state.preferences.language == UiLanguage::Spanish;
    state.dark = state.preferences.theme == ThemeMode::Dark;
    applyTheme(state);
    constexpr int label_x = 16;
    constexpr int edit_x = 146;
    constexpr int edit_width = 620;
    constexpr int row_y = 80;

    const HWND heading = createControl(state, 0, L"STATIC", kProductName, SS_LEFT, 16, 14, 400, 30, 0);
    SendMessageW(heading, WM_SETFONT, reinterpret_cast<WPARAM>(state.heading_font), TRUE);
    const HWND subtitle = createControl(
        state, 0, L"STATIC", L"Camera analytics control center", SS_LEFT, 18, 46, 400, 20, 0);
    addLocalizedText(state, subtitle, L"Camera analytics control center", L"Centro de control de analítica de cámaras");
    state.language = createControl(
        state, 0, L"BUTTON", L"Switch language to Spanish", WS_TABSTOP | BS_OWNERDRAW,
        810, 18, 34, 30, LanguageButton);
    state.theme_button = createControl(
        state, 0, L"BUTTON", L"Switch to dark theme", WS_TABSTOP | BS_OWNERDRAW,
        852, 18, 34, 30, ThemeButton);
    addTooltip(state, state.language, L"Language / Idioma");
    addTooltip(state, state.theme_button, L"Light or dark theme / Tema claro u oscuro");

    addLocalizedText(
        state, createLabel(state, L"Camera or video file", label_x, row_y, 120),
        L"Camera or video file", L"Cámara o archivo de video");
    state.source = createEdit(state, SourceEdit, edit_x, row_y, edit_width);
    SetWindowTextW(state.source, L"rtsp://");
    addLocalizedText(state, createBrowseButton(state, SourceBrowse, row_y), L"Browse...", L"Explorar...");

    addLocalizedText(
        state, createLabel(state, L"Camera ID", label_x, row_y + 36, 120),
        L"Camera ID", L"ID de cámara");
    state.source_label = createEdit(state, SourceLabelEdit, edit_x, row_y + 36, 430);
    const HWND load_env = createControl(
        state, 0, L"BUTTON", L"Load .env...", WS_TABSTOP | BS_OWNERDRAW,
        778, row_y + 36, 92, 25, LoadEnvButton);
    addLocalizedText(state, load_env, L"Load .env...", L"Cargar .env...");

    addLocalizedText(
        state, createLabel(state, L"Saved camera", label_x, row_y + 72, 120),
        L"Saved camera", L"Cámara guardada");
    state.saved_camera = createControl(
        state, 0, WC_COMBOBOXW, L"", WS_TABSTOP | CBS_DROPDOWNLIST,
        edit_x, row_y + 72, 430, 200, SavedCameraCombo);
    state.save_camera = createControl(
        state, 0, L"BUTTON", L"Save", WS_TABSTOP | BS_OWNERDRAW,
        586, row_y + 72, 80, 25, SaveCameraButton);
    addLocalizedText(state, state.save_camera, L"Save", L"Guardar");
    state.load_camera = createControl(
        state, 0, L"BUTTON", L"Load", WS_TABSTOP | BS_OWNERDRAW,
        674, row_y + 72, 80, 25, LoadCameraButton);
    addLocalizedText(state, state.load_camera, L"Load", L"Cargar");
    state.delete_camera = createControl(
        state, 0, L"BUTTON", L"Delete", WS_TABSTOP | BS_OWNERDRAW,
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
        state, createLabel(state, L"Inference size", 684, row_y + 146, 92),
        L"Inference size", L"Tamaño de inferencia");
    state.image_size = createClosedCombo(state, ImageSizeCombo, 778, row_y + 144, 92);
    for (const int image_size : kAllowedImageSizes) {
        const std::wstring value = std::to_wstring(image_size);
        SendMessageW(state.image_size, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(value.c_str()));
    }
    const auto image_position = std::ranges::find(kAllowedImageSizes, state.preferences.image_size);
    SendMessageW(state.image_size, CB_SETCURSEL,
        image_position == kAllowedImageSizes.end() ? 0 : image_position - kAllowedImageSizes.begin(), 0);

    const HWND threshold_heading = createLabel(
        state, L"PPE class confidence (0.00-1.00)", label_x, row_y + 184, 320);
    addLocalizedText(
        state, threshold_heading,
        L"PPE class confidence (0.00-1.00)", L"Confianza por clase EPP (0.00-1.00)");
    constexpr std::array<int, 8> threshold_x{146, 146, 146, 146, 620, 620, 620, 620};
    constexpr std::array<int, 8> label_positions{16, 16, 16, 16, 440, 440, 440, 440};
    constexpr std::array<std::size_t, cuajone::kPpeItemCount> item_class_ids{0, 2, 3, 4, 5, 6, 7};
    for (std::size_t index = 0; index < kPpeOutputLabels.size(); ++index) {
        const int row = static_cast<int>(index % 4);
        const int y = row_y + 214 + row * 34;
        const int text_length = MultiByteToWideChar(
            CP_UTF8, MB_ERR_INVALID_CHARS, kPpeOutputLabels[index].data(),
            static_cast<int>(kPpeOutputLabels[index].size()), nullptr, 0);
        std::wstring label(static_cast<std::size_t>(text_length), L'\0');
        MultiByteToWideChar(
            CP_UTF8, MB_ERR_INVALID_CHARS, kPpeOutputLabels[index].data(),
            static_cast<int>(kPpeOutputLabels[index].size()), label.data(), text_length);
        createLabel(state, label.c_str(), label_positions[index], y, 190);
        state.ppe_thresholds[index] = createThresholdCombo(
            state, PpeThresholdBase + static_cast<int>(index), threshold_x[index], y, 100);
        populateThresholdCombo(
            state.ppe_thresholds[index], state.preferences.ppe_class_confidences[index]);
        if (!SetWindowSubclass(
                state.ppe_thresholds[index], thresholdWheelProcedure,
                static_cast<UINT_PTR>(PpeThresholdBase + static_cast<int>(index)),
                reinterpret_cast<DWORD_PTR>(&state))) {
            throw std::runtime_error("Could not handle PPE threshold mouse-wheel input");
        }
        const HWND edit = GetWindow(state.ppe_thresholds[index], GW_CHILD);
        if (edit == nullptr || !SetWindowSubclass(
                edit, thresholdWheelProcedure,
                static_cast<UINT_PTR>(PpeThresholdBase + static_cast<int>(index)),
                reinterpret_cast<DWORD_PTR>(&state))) {
            throw std::runtime_error("Could not handle PPE threshold text input");
        }
        const auto item = std::ranges::find(item_class_ids, index);
        if (item != item_class_ids.end()) {
            const std::size_t item_index = static_cast<std::size_t>(item - item_class_ids.begin());
            state.ppe_enabled[item_index] = createControl(
                state, 0, L"BUTTON", label.c_str(), WS_TABSTOP | BS_AUTOCHECKBOX,
                threshold_x[index] + 100, y, index < 4 ? 180 : 150, 22,
                PpeEnabledBase + static_cast<int>(item_index));
            SendMessageW(state.ppe_enabled[item_index], BM_SETCHECK,
                state.preferences.ppe_enabled[item_index] ? BST_CHECKED : BST_UNCHECKED, 0);
        }
    }

    state.show = createControl(
        state, 0, L"BUTTON", L"Show annotated video window",
        WS_TABSTOP | BS_AUTOCHECKBOX, edit_x, row_y + 358, 340, 24, ShowCheck);
    SendMessageW(
        state.show, BM_SETCHECK,
        state.preferences.show_window ? BST_CHECKED : BST_UNCHECKED, 0);
    addLocalizedText(
        state, state.show, L"Show annotated video window", L"Mostrar ventana de video anotada");

    state.validate = createControl(
        state, 0, L"BUTTON", L"Validate", WS_TABSTOP | BS_OWNERDRAW,
        edit_x, row_y + 396, 110, 32, ValidateButton);
    addLocalizedText(state, state.validate, L"Validate", L"Validar");
    state.start = createControl(
        state, 0, L"BUTTON", L"Start", WS_TABSTOP | BS_OWNERDRAW,
        270, row_y + 396, 110, 32, StartButton);
    addLocalizedText(state, state.start, L"Start", L"Iniciar");
    state.stop = createControl(
        state, 0, L"BUTTON", L"Stop", WS_TABSTOP | BS_OWNERDRAW,
        394, row_y + 396, 110, 32, StopButton);
    addLocalizedText(state, state.stop, L"Stop", L"Detener");
    EnableWindow(state.stop, FALSE);

    addLocalizedText(
        state, createLabel(state, L"Status", label_x, row_y + 448, 120),
        L"Status", L"Estado");
    state.status = createControl(
        state, WS_EX_CLIENTEDGE, L"STATIC", L"Ready", SS_LEFT | SS_CENTERIMAGE,
        edit_x, row_y + 444, 724, 32, StatusText);
    addLocalizedText(
        state, createLabel(state, L"Log path", label_x, row_y + 492, 120),
        L"Log path", L"Ruta del log");
    state.log_path = createEdit(state, LogPathEdit, edit_x, row_y + 488, 620, true);
    const HWND open_log = createControl(
        state, 0, L"BUTTON", L"Open log", WS_TABSTOP | BS_OWNERDRAW,
        778, row_y + 488, 92, 25, OpenLogButton);
    addLocalizedText(state, open_log, L"Open log", L"Abrir log");

    state.program_data = knownProgramData();
    state.managed_model_root = preferredModelRoot();
    setText(state.output, state.program_data / L"output");
    SetWindowTextW(state.source_label, L"CAM_CUAJONE_01");
    setText(state.log_path, state.program_data / L"logs");
    refreshSavedCameraProfiles(state);

    if (!resolveManagedModelSet(state.managed_model_root, true).onnx_complete) {
        setStatus(
            state,
            state.spanish
                ? L"El conjunto obligatorio de modelos administrados está incompleto en la carpeta de la aplicación."
                : L"The mandatory managed model set is incomplete at the application models folder.");
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
    if (const auto confidence = envValue(values, L"PPE_CONF")) {
        int selection{};
        try {
            selection = static_cast<int>(std::lround(parsePpeConfidenceThreshold(*confidence) * 100.0F));
        } catch (const std::invalid_argument&) {
            throw std::invalid_argument("PPE_CONF must be a decimal from 0.00 to 1.00");
        }
        for (HWND threshold : state.ppe_thresholds) {
            setThresholdComboValue(threshold, selection);
        }
    }
    const auto ppe_imgsz = envValue(values, L"PPE_IMGSZ");
    const auto pose_imgsz = envValue(values, L"POSE_IMGSZ");
    if (ppe_imgsz || pose_imgsz) {
        if (ppe_imgsz && pose_imgsz && *ppe_imgsz != *pose_imgsz) {
            throw std::invalid_argument("PPE_IMGSZ and POSE_IMGSZ must match");
        }
        const std::wstring& configured = ppe_imgsz ? *ppe_imgsz : *pose_imgsz;
        wchar_t* end = nullptr;
        const long parsed = std::wcstol(configured.c_str(), &end, 10);
        const auto found = std::ranges::find(kAllowedImageSizes, static_cast<int>(parsed));
        if (end != configured.c_str() + configured.size() || found == kAllowedImageSizes.end()) {
            throw std::invalid_argument("PPE_IMGSZ/POSE_IMGSZ must be 640, 768, 960, or 1280");
        }
        SendMessageW(state.image_size, CB_SETCURSEL, found - kAllowedImageSizes.begin(), 0);
    }

    std::vector<std::wstring> ignored;
    state.runtime_options.clear();
    const auto append = [&](std::wstring_view key, std::wstring_view option) {
        if (const auto value = envValue(values, key)) {
            state.runtime_options.emplace_back(std::wstring(option), *value);
        }
    };
    append(L"TARGET_INFERENCE_FPS", L"--target-fps");
    append(L"POSE_CONF", L"--pose-conf");
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
             L"EXCEL_EXPORT_EVERY_EVENTS",
             L"PPE_MODEL_PATH", L"POSE_MODEL_PATH", L"RTSP_SOCKET_TIMEOUT_S"}) {
        if (envValue(values, key)) ignored.emplace_back(key);
    }
    if (ignored.empty()) {
        setStatus(state, state.spanish ? L"Configuración .env cargada" : L"Loaded .env settings");
    } else {
        setStatus(
            state,
            state.spanish
                ? L".env cargado; se ignoraron opciones exclusivas de Python"
                : L"Loaded .env; Python-only settings were ignored");
    }
    persistPreferences(state);
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

std::filesystem::path pickVideoFile(HWND owner) {
    IFileOpenDialog* dialog = nullptr;
    HRESULT result = CoCreateInstance(
        CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER,
        IID_PPV_ARGS(&dialog));
    if (FAILED(result)) return {};
    DWORD options{};
    dialog->GetOptions(&options);
    dialog->SetOptions(options | FOS_FORCEFILESYSTEM | FOS_PATHMUSTEXIST | FOS_FILEMUSTEXIST);
    const COMDLG_FILTERSPEC filters[] = {
        {L"Video files", L"*.mp4;*.avi;*.mov;*.mkv"},
        {L"All files", L"*.*"},
    };
    dialog->SetFileTypes(static_cast<UINT>(std::size(filters)), filters);
    dialog->SetFileTypeIndex(1);
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
    settings.managed_model_root = state.managed_model_root;
    settings.source_label = editText(state.source_label);
    settings.runtime_options = state.runtime_options;
    const LRESULT image_selection = SendMessageW(state.image_size, CB_GETCURSEL, 0, 0);
    if (image_selection == CB_ERR
        || static_cast<std::size_t>(image_selection) >= kAllowedImageSizes.size()) {
        throw std::invalid_argument("Select a supported inference size");
    }
    settings.image_size = kAllowedImageSizes[static_cast<std::size_t>(image_selection)];
    for (std::size_t index = 0; index < settings.ppe_class_confidences.size(); ++index) {
        settings.ppe_class_confidences[index] = parsePpeConfidenceThreshold(
            editText(state.ppe_thresholds[index]));
    }
    for (std::size_t index = 0; index < settings.ppe_enabled.size(); ++index) {
        settings.ppe_enabled[index] = SendMessageW(
            state.ppe_enabled[index], BM_GETCHECK, 0, 0) == BST_CHECKED;
    }
    settings.show_window = SendMessageW(state.show, BM_GETCHECK, 0, 0) == BST_CHECKED;
    return settings;
}

void persistPreferences(LauncherWindow& state) {
    const LauncherSettings current = readSettings(state);
    state.preferences.language = state.spanish ? UiLanguage::Spanish : UiLanguage::English;
    state.preferences.theme = state.dark ? ThemeMode::Dark : ThemeMode::Light;
    state.preferences.image_size = current.image_size;
    state.preferences.ppe_class_confidences = current.ppe_class_confidences;
    state.preferences.ppe_enabled = current.ppe_enabled;
    state.preferences.show_window = current.show_window;
    for (std::size_t index = 0; index < current.ppe_class_confidences.size(); ++index) {
        setThresholdComboValue(
            state.ppe_thresholds[index],
            static_cast<int>(std::lround(current.ppe_class_confidences[index] * 100.0F)));
    }
    saveOperatorPreferencesAtomic(state.preferences_path, state.preferences);
}

std::filesystem::path siblingRuntime() {
    std::wstring module_path(32768, L'\0');
    const DWORD length = GetModuleFileNameW(
        nullptr, module_path.data(), static_cast<DWORD>(module_path.size()));
    if (length == 0 || length == static_cast<DWORD>(module_path.size())) {
        throw std::runtime_error("Could not resolve the launcher executable path");
    }
    module_path.resize(length);
    return std::filesystem::path(module_path).parent_path() / kRuntimeExecutable;
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
        throw std::runtime_error("Sibling " + runtimeExecutableName() + " was not found");
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
        throw std::runtime_error("CreateProcessW failed for " + runtimeExecutableName());
    }
    if (!AssignProcessToJobObject(job, process.hProcess)) {
        TerminateProcess(process.hProcess, 1);
        CloseHandle(process.hThread);
        CloseHandle(process.hProcess);
        CloseHandle(pipe_read);
        CloseHandle(log);
        CloseHandle(job);
        throw std::runtime_error("Could not assign " + runtimeExecutableName() + " to its Job Object");
    }
    if (ResumeThread(process.hThread) == static_cast<DWORD>(-1)) {
        TerminateJobObject(job, 1);
        CloseHandle(process.hThread);
        CloseHandle(process.hProcess);
        CloseHandle(pipe_read);
        CloseHandle(log);
        CloseHandle(job);
        throw std::runtime_error("Could not resume " + runtimeExecutableName());
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
    setStatus(
        state,
        preflight
            ? (state.spanish ? L"Validando configuración..." : L"Validating configuration...")
            : (state.spanish ? L"El runtime está en ejecución" : L"Runtime is running"));
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
            ? (state.spanish
                ? L"Deteniendo de forma segura; se forzará después de 30 segundos"
                : L"Stopping gracefully; forced termination follows after 30 seconds")
            : (state.spanish
                ? L"Falló CTRL_BREAK; se forzará la detención después de 30 segundos"
                : L"CTRL_BREAK delivery failed; forced termination follows after 30 seconds"));
}

void finishProcess(LauncherWindow& state, DWORD exit_code, bool preflight) {
    const bool stopped = state.stop_requested.load(std::memory_order_relaxed);
    if (state.waiter.joinable()) state.waiter.join();
    if (state.output_pump.joinable()) state.output_pump.join();
    closeProcessHandles(state);
    setRunning(state, false);
    if (stopped) {
        setStatus(
            state,
            (state.spanish ? L"Runtime detenido (código de salida " : L"Runtime stopped (exit code ")
                + std::to_wstring(exit_code) + L")");
    } else if (exit_code == 0) {
        setStatus(
            state,
            preflight
                ? (state.spanish ? L"Validación aprobada" : L"Validation passed")
                : (state.spanish ? L"Runtime completado correctamente" : L"Runtime completed successfully"));
    } else {
        setStatus(
            state,
            (preflight
                ? (state.spanish ? L"Falló la validación (código de salida " : L"Validation failed (exit code ")
                : (state.spanish ? L"Falló el runtime (código de salida " : L"Runtime failed (exit code "))
                + std::to_wstring(exit_code)
                + (state.spanish ? L"); consulta el log" : L"); see log"));
    }
    state.stop_requested.store(false, std::memory_order_relaxed);
    if (state.close_requested) DestroyWindow(state.window);
}

void showError(LauncherWindow& state, const std::exception& error) {
    const std::string narrow(error.what());
    const std::wstring message(narrow.begin(), narrow.end());
    setStatus(state, state.spanish ? L"Error de configuración" : L"Configuration error");
    MessageBoxW(state.window, message.c_str(), kProductName, MB_OK | MB_ICONERROR);
}

LRESULT CALLBACK thresholdWheelProcedure(
    HWND window, UINT message, WPARAM wparam, LPARAM lparam, UINT_PTR, DWORD_PTR reference) {
    if (message != WM_MOUSEWHEEL) return DefSubclassProc(window, message, wparam, lparam);

    auto& state = *reinterpret_cast<LauncherWindow*>(reference);
    HWND combo = GetParent(window) == state.window ? window : GetParent(window);
    const auto threshold = std::ranges::find(state.ppe_thresholds, combo);
    if (threshold == state.ppe_thresholds.end()) {
        return DefSubclassProc(window, message, wparam, lparam);
    }
    const int steps = GET_WHEEL_DELTA_WPARAM(wparam) / WHEEL_DELTA;
    if (steps == 0) return 0;
    try {
        const int current = static_cast<int>(std::lround(
            parsePpeConfidenceThreshold(editText(combo)) * 100.0F));
        SetFocus(combo);
        setThresholdComboValue(combo, current + steps * 2);
        persistPreferences(state);
    } catch (const std::exception& error) {
        showError(state, error);
    }
    return 0;
}

void browseInto(LauncherWindow& state, int id) {
    if (id == SourceBrowse) {
        const auto path = pickVideoFile(state.window);
        if (!path.empty()) setText(state.source, path);
    } else if (id == OutputBrowse) {
        const auto path = pickFolder(state.window);
        if (!path.empty()) setText(state.output, path);
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
            case WM_CTLCOLORLISTBOX: {
                const Palette colors = palette(*state);
                SetTextColor(reinterpret_cast<HDC>(wparam), colors.text);
                SetBkColor(reinterpret_cast<HDC>(wparam), colors.input);
                return reinterpret_cast<LRESULT>(state->input_brush);
            }
            case WM_CTLCOLORSTATIC:
                {
                const Palette colors = palette(*state);
                if (reinterpret_cast<HWND>(lparam) == state->status) {
                    SetTextColor(reinterpret_cast<HDC>(wparam), state->dark ? colors.text : colors.primary);
                    SetBkColor(reinterpret_cast<HDC>(wparam), colors.status);
                    return reinterpret_cast<LRESULT>(state->status_brush);
                }
                SetTextColor(reinterpret_cast<HDC>(wparam), colors.muted);
                SetBkMode(reinterpret_cast<HDC>(wparam), TRANSPARENT);
                return reinterpret_cast<LRESULT>(state->window_brush);
                }
            case WM_CTLCOLORBTN:
                SetTextColor(reinterpret_cast<HDC>(wparam), palette(*state).text);
                SetBkMode(reinterpret_cast<HDC>(wparam), TRANSPARENT);
                return reinterpret_cast<LRESULT>(state->window_brush);
            case WM_DRAWITEM:
                if (const auto* item = reinterpret_cast<const DRAWITEMSTRUCT*>(lparam);
                    item->CtlType == ODT_BUTTON) {
                    drawButton(*state, *item);
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
                else if (id == LanguageButton) {
                    state->spanish = !state->spanish;
                    refreshLanguage(*state);
                    persistPreferences(*state);
                }
                else if (id == ThemeButton) {
                    state->dark = !state->dark;
                    applyTheme(*state);
                    refreshLanguage(*state);
                    persistPreferences(*state);
                }
                else if (id == LoadEnvButton) loadEnv(*state);
                else if (id == SourceBrowse || id == OutputBrowse) {
                    browseInto(*state, id);
                }
                else if (id == ShowCheck && HIWORD(wparam) == BN_CLICKED) {
                    persistPreferences(*state);
                }
                else if (id >= PpeEnabledBase
                    && id < PpeEnabledBase + static_cast<int>(cuajone::kPpeItemCount)
                    && HIWORD(wparam) == BN_CLICKED) {
                    persistPreferences(*state);
                }
                else if (id == ImageSizeCombo && HIWORD(wparam) == CBN_SELCHANGE) {
                    persistPreferences(*state);
                }
                else if (id >= PpeThresholdBase
                    && id < PpeThresholdBase + static_cast<int>(kPpeOutputLabels.size())
                    && (HIWORD(wparam) == CBN_SELCHANGE || HIWORD(wparam) == CBN_KILLFOCUS)) {
                    persistPreferences(*state);
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
    INITCOMMONCONTROLSEX common_controls{
        sizeof(common_controls), ICC_STANDARD_CLASSES | ICC_WIN95_CLASSES};
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
    try {
        state.preferences_path = knownLocalAppData();
        state.preferences = loadOperatorPreferences(state.preferences_path);
    } catch (const std::exception&) {
        state.preferences = {};
    }
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
