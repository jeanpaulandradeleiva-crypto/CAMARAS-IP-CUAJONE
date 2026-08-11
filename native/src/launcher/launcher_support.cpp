// SPDX-License-Identifier: AGPL-3.0-only

#include "cuajone/launcher_support.hpp"

#define NOMINMAX

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cwctype>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <windows.h>
#include <stdexcept>

namespace cuajone::launcher {
namespace {

constexpr wchar_t kSavedCameraCredentialTargetPrefix[] = L"NexoAI Vision/RTSP/";

bool regularFile(const std::filesystem::path& path) {
    std::error_code error;
    return !path.empty() && std::filesystem::is_regular_file(path, error) && !error;
}

bool regularFileWithExtension(
    const std::filesystem::path& path,
    std::wstring_view expected_extension) {
    if (!regularFile(path)) return false;
    std::wstring extension = path.extension().wstring();
    std::transform(extension.begin(), extension.end(), extension.begin(), [](wchar_t character) {
        return static_cast<wchar_t>(std::towlower(character));
    });
    return extension == expected_extension;
}

std::wstring wideFromUtf8(std::string_view value) {
    if (value.empty()) return {};
    const int required = MultiByteToWideChar(
        CP_UTF8, MB_ERR_INVALID_CHARS, value.data(), static_cast<int>(value.size()), nullptr, 0);
    if (required <= 0) throw std::invalid_argument("Preferences are not valid UTF-8");
    std::wstring result(static_cast<std::size_t>(required), L'\0');
    if (MultiByteToWideChar(
            CP_UTF8, MB_ERR_INVALID_CHARS, value.data(), static_cast<int>(value.size()),
            result.data(), required) <= 0) {
        throw std::invalid_argument("Preferences are not valid UTF-8");
    }
    return result;
}

bool isRtspSource(std::wstring_view source) {
    return source.starts_with(L"rtsp://") || source.starts_with(L"rtsps://");
}

bool isSavedCameraProfileCharacter(wchar_t character) {
    return std::iswalnum(character) != 0 || character == L' ' || character == L'_'
        || character == L'-' || character == L'.';
}

}  // namespace

bool isValidSavedCameraProfileName(std::wstring_view name) {
    return !name.empty() && name.size() <= 80
        && std::all_of(name.begin(), name.end(), isSavedCameraProfileCharacter);
}

std::wstring_view savedCameraCredentialTargetPrefix() {
    return kSavedCameraCredentialTargetPrefix;
}

std::wstring savedCameraCredentialTarget(std::wstring_view name) {
    if (!isValidSavedCameraProfileName(name)) {
        throw std::invalid_argument("Saved camera profile name is invalid");
    }
    return std::wstring(kSavedCameraCredentialTargetPrefix) + std::wstring(name);
}

void validateRtspCameraUrl(std::wstring_view source) {
    if (!isRtspSource(source)) {
        throw std::invalid_argument("Source must be an rtsp:// or rtsps:// camera URL");
    }
    const std::size_t scheme_end = source.find(L"://");
    const std::size_t authority_start = scheme_end + 3;
    const std::size_t authority_end = source.find_first_of(L"/?#", authority_start);
    const std::size_t end = authority_end == std::wstring_view::npos ? source.size() : authority_end;
    if (authority_start >= end) {
        throw std::invalid_argument("RTSP camera URL must include a host");
    }
    const std::wstring_view authority = source.substr(authority_start, end - authority_start);
    if (std::any_of(authority.begin(), authority.end(), [](wchar_t character) {
            return std::iswspace(character) != 0 || character < 0x20;
        })) {
        throw std::invalid_argument("RTSP camera URL authority is invalid");
    }
    const std::size_t at = authority.rfind(L'@');
    const std::wstring_view host_and_port = at == std::wstring_view::npos
        ? authority : authority.substr(at + 1);
    if (host_and_port.empty()) {
        throw std::invalid_argument("RTSP camera URL must include a host");
    }
}

namespace {

bool hasNonWhitespace(std::wstring_view value) {
    return std::any_of(value.begin(), value.end(), [](wchar_t character) {
        return std::iswspace(character) == 0;
    });
}

std::string utf8FromWide(std::wstring_view value) {
    if (value.empty()) return {};
    const int required = WideCharToMultiByte(
        CP_UTF8, 0, value.data(), static_cast<int>(value.size()), nullptr, 0, nullptr, nullptr);
    if (required <= 0) return {};
    std::string result(static_cast<std::size_t>(required), '\0');
    const int written = WideCharToMultiByte(
        CP_UTF8, 0, value.data(), static_cast<int>(value.size()),
        result.data(), required, nullptr, nullptr);
    if (written <= 0) return {};
    return result;
}

void appendOption(
    std::vector<std::wstring>& arguments,
    std::wstring_view option,
    const std::filesystem::path& value) {
    arguments.emplace_back(option);
    arguments.push_back(value.wstring());
}

}  // namespace

std::filesystem::path adjacentOnnxManifest(const std::filesystem::path& model) {
    std::filesystem::path result = model;
    result += L".manifest.json";
    return result;
}

ManagedModelSet resolveManagedModelSet(
    const std::filesystem::path& root,
    bool pose_required) {
    ManagedModelSet result;
    result.root = root;
    result.ppe_engine = root / L"ppe.engine";
    result.pose_engine = root / L"pose.engine";
    result.ppe_onnx = root / L"ppe.onnx";
    result.pose_onnx = root / L"pose.onnx";
    result.tensor_rt_complete = regularFileWithExtension(result.ppe_engine, L".engine")
        && (!pose_required || regularFileWithExtension(result.pose_engine, L".engine"));
    result.onnx_complete = regularFileWithExtension(result.ppe_onnx, L".onnx")
        && regularFile(adjacentOnnxManifest(result.ppe_onnx))
        && (!pose_required || (regularFileWithExtension(result.pose_onnx, L".onnx")
            && regularFile(adjacentOnnxManifest(result.pose_onnx))));
    return result;
}

LaunchPlan buildLaunchPlan(const LauncherSettings& settings, bool preflight) {
    if (settings.source.empty()) {
        throw std::invalid_argument("RTSP camera URL is required");
    }
    validateRtspCameraUrl(settings.source);
    if (settings.output.empty()) {
        throw std::invalid_argument("Output folder is required");
    }
    std::error_code output_error;
    if (std::filesystem::exists(settings.output, output_error)
        && !std::filesystem::is_directory(settings.output, output_error)) {
        throw std::invalid_argument("Output path must be a folder");
    }

    validateImageSize(settings.image_size);
    validatePpeClassConfidences(settings.ppe_class_confidences);
    if (settings.managed_model_root.empty()) {
        throw std::invalid_argument("Managed model location is required");
    }
    for (const auto& [option, value] : settings.runtime_options) {
        static_cast<void>(value);
        if (option == L"--ppe-engine" || option == L"--pose-engine"
            || option == L"--ppe-onnx" || option == L"--pose-onnx"
            || option == L"--ppe-labels") {
            throw std::invalid_argument("Model paths and labels cannot be supplied through launcher options");
        }
    }

    const bool needs_pose = settings.analytics_mode == AnalyticsMode::PpeFall;
    const ManagedModelSet models = resolveManagedModelSet(settings.managed_model_root, needs_pose);
    const bool tensor_rt_candidate = models.tensor_rt_complete;
    const bool cpu_candidate = models.onnx_complete;

    const bool cuda_candidate = tensor_rt_candidate || cpu_candidate;
    if (settings.compute_mode == ComputeMode::Cuda && !cuda_candidate) {
        std::wstring message = L"The complete managed CUDA model set is missing at ";
        message += settings.managed_model_root.wstring();
        throw std::invalid_argument(
            utf8FromWide(message));
    }
    if (settings.compute_mode == ComputeMode::Cpu && !cpu_candidate) {
        std::wstring message = L"The complete managed ONNX model set is missing at ";
        message += settings.managed_model_root.wstring();
        throw std::invalid_argument(
            utf8FromWide(message));
    }
    if (settings.compute_mode == ComputeMode::Auto && !cuda_candidate && !cpu_candidate) {
        std::wstring message = L"No complete managed model set was found at ";
        message += settings.managed_model_root.wstring();
        throw std::invalid_argument(
            utf8FromWide(message));
    }

    LaunchPlan result;
    result.has_cuda_candidate = cuda_candidate;
    result.has_cpu_candidate = cpu_candidate;
    if (preflight) result.arguments.emplace_back(L"--preflight");
    result.arguments.emplace_back(L"--source");
    result.arguments.push_back(settings.source);
    if (hasNonWhitespace(settings.source_label)) {
        result.arguments.emplace_back(L"--source-label");
        result.arguments.push_back(settings.source_label);
    }
    appendOption(result.arguments, L"--output", settings.output);
    result.arguments.emplace_back(L"--mode");
    result.arguments.emplace_back(needs_pose ? L"ppe-fall" : L"ppe-only");
    result.arguments.emplace_back(L"--compute");
    switch (settings.compute_mode) {
        case ComputeMode::Auto: result.arguments.emplace_back(L"auto"); break;
        case ComputeMode::Cuda: result.arguments.emplace_back(L"cuda"); break;
        case ComputeMode::Cpu: result.arguments.emplace_back(L"cpu"); break;
    }
    result.arguments.emplace_back(L"--imgsz");
    result.arguments.push_back(std::to_wstring(settings.image_size));
    for (std::size_t index = 0; index < kPpeOutputLabels.size(); ++index) {
        result.arguments.emplace_back(L"--ppe-class-conf");
        std::wostringstream value;
        value << wideFromUtf8(kPpeOutputLabels[index]) << L'=' << std::fixed
              << std::setprecision(2) << settings.ppe_class_confidences[index];
        result.arguments.push_back(value.str());
    }

    const bool include_tensor_rt = tensor_rt_candidate && settings.compute_mode != ComputeMode::Cpu;
    const bool include_onnx = cpu_candidate;
    if (include_tensor_rt) {
        appendOption(result.arguments, L"--ppe-engine", models.ppe_engine);
        if (needs_pose) appendOption(result.arguments, L"--pose-engine", models.pose_engine);
    }
    if (include_onnx) {
        appendOption(result.arguments, L"--ppe-onnx", models.ppe_onnx);
        if (needs_pose) appendOption(result.arguments, L"--pose-onnx", models.pose_onnx);
    }
    result.arguments.emplace_back(L"--ppe-labels");
    result.arguments.emplace_back(
        L"Gloves,Person,Safety_boots,Vest,respirador,tapaorejas,Hard_hat,lentes_protectores");
    for (const auto& [option, value] : settings.runtime_options) {
        if (!option.empty() && hasNonWhitespace(value)) {
            result.arguments.push_back(option);
            result.arguments.push_back(value);
        }
    }
    if (settings.show_window) result.arguments.emplace_back(L"--show");
    return result;
}

OperatorPreferences parseOperatorPreferences(std::string_view text) {
    std::map<std::string, std::string> values;
    std::istringstream input{std::string(text)};
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        const auto separator = line.find('=');
        if (separator == std::string::npos || separator == 0
            || !values.emplace(line.substr(0, separator), line.substr(separator + 1)).second) {
            throw std::invalid_argument("Preferences contain an invalid or duplicate entry");
        }
    }
    if (values.size() != 5 || !values.contains("schema_version")
        || !values.contains("language") || !values.contains("theme")
        || !values.contains("imgsz") || !values.contains("ppe_class_conf")) {
        throw std::invalid_argument("Preferences contain missing or unsupported entries");
    }
    OperatorPreferences result;
    if (values.at("schema_version") != "1") {
        throw std::invalid_argument("Unsupported preferences schema_version");
    }
    if (values.at("language") == "en") result.language = UiLanguage::English;
    else if (values.at("language") == "es") result.language = UiLanguage::Spanish;
    else throw std::invalid_argument("Preferences language must be en or es");
    if (values.at("theme") == "light") result.theme = ThemeMode::Light;
    else if (values.at("theme") == "dark") result.theme = ThemeMode::Dark;
    else throw std::invalid_argument("Preferences theme must be light or dark");
    const auto& image_text = values.at("imgsz");
    const auto image_result = std::from_chars(
        image_text.data(), image_text.data() + image_text.size(), result.image_size);
    if (image_result.ec != std::errc{} || image_result.ptr != image_text.data() + image_text.size()) {
        throw std::invalid_argument("Preferences imgsz is invalid");
    }
    validateImageSize(result.image_size);

    std::istringstream thresholds(values.at("ppe_class_conf"));
    std::string entry;
    std::size_t index = 0;
    while (std::getline(thresholds, entry, ',')) {
        if (index >= kPpeOutputLabels.size()) {
            throw std::invalid_argument("Preferences contain too many PPE thresholds");
        }
        const auto separator = entry.find(':');
        if (separator == std::string::npos
            || entry.substr(0, separator) != kPpeOutputLabels[index]) {
            throw std::invalid_argument("Preferences PPE threshold order is invalid");
        }
        const std::string_view number(entry.data() + separator + 1, entry.size() - separator - 1);
        const auto parsed = std::from_chars(
            number.data(), number.data() + number.size(), result.ppe_class_confidences[index]);
        if (parsed.ec != std::errc{} || parsed.ptr != number.data() + number.size()) {
            throw std::invalid_argument("Preferences PPE threshold value is invalid");
        }
        ++index;
    }
    if (index != kPpeOutputLabels.size()) {
        throw std::invalid_argument("Preferences require exactly eight PPE thresholds");
    }
    validatePpeClassConfidences(result.ppe_class_confidences);
    return result;
}

std::string serializeOperatorPreferences(const OperatorPreferences& preferences) {
    if (preferences.schema_version != 1) {
        throw std::invalid_argument("Unsupported preferences schema_version");
    }
    validateImageSize(preferences.image_size);
    validatePpeClassConfidences(preferences.ppe_class_confidences);
    std::ostringstream output;
    output << "schema_version=1\n"
           << "language=" << (preferences.language == UiLanguage::Spanish ? "es" : "en") << '\n'
           << "theme=" << (preferences.theme == ThemeMode::Dark ? "dark" : "light") << '\n'
           << "imgsz=" << preferences.image_size << '\n'
           << "ppe_class_conf=";
    for (std::size_t index = 0; index < kPpeOutputLabels.size(); ++index) {
        if (index != 0) output << ',';
        output << kPpeOutputLabels[index] << ':' << std::fixed << std::setprecision(2)
               << preferences.ppe_class_confidences[index];
    }
    output << '\n';
    return output.str();
}

OperatorPreferences loadOperatorPreferences(const std::filesystem::path& path) noexcept {
    try {
        std::ifstream input(path, std::ios::binary);
        if (!input) return {};
        const std::string text{
            std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
        if (text.size() > 16U * 1024U) return {};
        return parseOperatorPreferences(text);
    } catch (...) {
        return {};
    }
}

void saveOperatorPreferencesAtomic(
    const std::filesystem::path& path,
    const OperatorPreferences& preferences) {
    const std::string text = serializeOperatorPreferences(preferences);
    std::filesystem::create_directories(path.parent_path());
    std::filesystem::path temporary = path;
    temporary += L".tmp";
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        output.write(text.data(), static_cast<std::streamsize>(text.size()));
        output.flush();
        if (!output) throw std::runtime_error("Could not write operator preferences");
    }
    if (!MoveFileExW(
            temporary.c_str(), path.c_str(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw std::runtime_error("Could not atomically replace operator preferences");
    }
}

std::vector<std::string_view> visibleLauncherControlKeys() {
    return {
        "source", "source_label", "saved_camera", "output", "analytics", "compute",
        "imgsz", "Gloves", "Person", "Safety_boots", "Vest", "respirador",
        "tapaorejas", "Hard_hat", "lentes_protectores", "show", "language_icon",
        "theme_icon", "validate", "start", "stop", "status", "log_path",
    };
}

std::wstring quoteWindowsArgument(std::wstring_view argument) {
    if (argument.empty()) return L"\"\"";
    if (argument.find_first_of(L" \t\"") == std::wstring_view::npos) {
        return std::wstring(argument);
    }

    std::wstring result(1, L'\"');
    std::size_t backslashes = 0;
    for (const wchar_t character : argument) {
        if (character == L'\\') {
            ++backslashes;
            continue;
        }
        if (character == L'\"') {
            result.append(backslashes * 2 + 1, L'\\');
            result.push_back(L'\"');
        } else {
            result.append(backslashes, L'\\');
            result.push_back(character);
        }
        backslashes = 0;
    }
    result.append(backslashes * 2, L'\\');
    result.push_back(L'\"');
    return result;
}

std::wstring buildWindowsCommandLine(const std::vector<std::wstring>& arguments) {
    std::wstring result;
    for (const auto& argument : arguments) {
        if (!result.empty()) result.push_back(L' ');
        result += quoteWindowsArgument(argument);
    }
    return result;
}

std::string redactRtspCredentials(std::string_view text) {
    std::string result;
    std::size_t position = 0;
    while (position < text.size()) {
        const std::size_t rtsp = text.find("rtsp://", position);
        const std::size_t rtsps = text.find("rtsps://", position);
        const std::size_t scheme = rtsp == std::string_view::npos
            ? rtsps
            : (rtsps == std::string_view::npos ? rtsp : std::min(rtsp, rtsps));
        if (scheme == std::string_view::npos) {
            result.append(text.substr(position));
            break;
        }
        result.append(text.substr(position, scheme - position));
        const std::size_t scheme_end = text.find("://", scheme) + 3;
        const std::size_t authority_end = text.find_first_of("/?# \t\r\n", scheme_end);
        const std::size_t end = authority_end == std::string_view::npos
            ? text.size() : authority_end;
        const std::size_t at = text.substr(scheme_end, end - scheme_end).rfind('@');
        if (at == std::string_view::npos) {
            result.append(text.substr(scheme, end - scheme));
        } else {
            result.append(text.substr(scheme, scheme_end - scheme));
            result += "***@";
            result.append(text.substr(scheme_end + at + 1, end - scheme_end - at - 1));
        }
        position = end;
    }
    return result;
}

}  // namespace cuajone::launcher
