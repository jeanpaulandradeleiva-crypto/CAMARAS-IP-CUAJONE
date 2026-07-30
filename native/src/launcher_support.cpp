// SPDX-License-Identifier: AGPL-3.0-only

#include "cuajone/launcher_support.hpp"

#include <algorithm>
#include <cwctype>
#include <stdexcept>

namespace cuajone::launcher {
namespace {

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

bool isRtspSource(std::wstring_view source) {
    return source.starts_with(L"rtsp://") || source.starts_with(L"rtsps://");
}

bool hasNonWhitespace(std::wstring_view value) {
    return std::any_of(value.begin(), value.end(), [](wchar_t character) {
        return std::iswspace(character) == 0;
    });
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

LaunchPlan buildLaunchPlan(const LauncherSettings& settings, bool preflight) {
    if (settings.source.empty()) {
        throw std::invalid_argument("Source is required");
    }
    if (!isRtspSource(settings.source) && !regularFile(std::filesystem::path(settings.source))) {
        throw std::invalid_argument("Offline source must be an existing file");
    }
    if (settings.output.empty()) {
        throw std::invalid_argument("Output folder is required");
    }
    std::error_code output_error;
    if (std::filesystem::exists(settings.output, output_error)
        && !std::filesystem::is_directory(settings.output, output_error)) {
        throw std::invalid_argument("Output path must be a folder");
    }

    const bool needs_pose = settings.analytics_mode == AnalyticsMode::PpeFall;
    const bool cuda_candidate = regularFileWithExtension(settings.ppe_engine, L".engine")
        && (!needs_pose || regularFileWithExtension(settings.pose_engine, L".engine"));
    const bool cpu_candidate = regularFileWithExtension(settings.ppe_onnx, L".onnx")
        && regularFile(adjacentOnnxManifest(settings.ppe_onnx))
        && hasNonWhitespace(settings.ppe_labels)
        && (!needs_pose || (regularFileWithExtension(settings.pose_onnx, L".onnx")
            && regularFile(adjacentOnnxManifest(settings.pose_onnx))));

    if (settings.compute_mode == ComputeMode::Cuda && !cuda_candidate) {
        throw std::invalid_argument(
            needs_pose
                ? "CUDA requires existing PPE and pose engines in PPE+fall mode"
                : "CUDA requires an existing PPE engine");
    }
    if (settings.compute_mode == ComputeMode::Cpu && !cpu_candidate) {
        throw std::invalid_argument(
            needs_pose
                ? "CPU requires PPE and pose ONNX files, adjacent manifests, and PPE labels in PPE+fall mode"
                : "CPU requires a PPE ONNX file, adjacent manifest, and PPE labels");
    }
    if (settings.compute_mode == ComputeMode::Auto && !cuda_candidate && !cpu_candidate) {
        throw std::invalid_argument(
            "Auto requires a complete CUDA candidate, a complete CPU candidate, or both");
    }

    LaunchPlan result;
    result.has_cuda_candidate = cuda_candidate;
    result.has_cpu_candidate = cpu_candidate;
    if (preflight) result.arguments.emplace_back(L"--preflight");
    result.arguments.emplace_back(L"--source");
    result.arguments.push_back(settings.source);
    appendOption(result.arguments, L"--output", settings.output);
    result.arguments.emplace_back(L"--mode");
    result.arguments.emplace_back(needs_pose ? L"ppe-fall" : L"ppe-only");
    result.arguments.emplace_back(L"--compute");
    switch (settings.compute_mode) {
        case ComputeMode::Auto: result.arguments.emplace_back(L"auto"); break;
        case ComputeMode::Cuda: result.arguments.emplace_back(L"cuda"); break;
        case ComputeMode::Cpu: result.arguments.emplace_back(L"cpu"); break;
    }

    const bool include_cuda = cuda_candidate && settings.compute_mode != ComputeMode::Cpu;
    const bool include_cpu = cpu_candidate && settings.compute_mode != ComputeMode::Cuda;
    if (include_cuda) {
        appendOption(result.arguments, L"--ppe-engine", settings.ppe_engine);
        if (needs_pose) appendOption(result.arguments, L"--pose-engine", settings.pose_engine);
    }
    if (include_cpu) {
        appendOption(result.arguments, L"--ppe-onnx", settings.ppe_onnx);
        if (needs_pose) appendOption(result.arguments, L"--pose-onnx", settings.pose_onnx);
    }
    if (hasNonWhitespace(settings.ppe_labels)) {
        result.arguments.emplace_back(L"--ppe-labels");
        result.arguments.push_back(settings.ppe_labels);
    }
    if (settings.show_window) result.arguments.emplace_back(L"--show");
    return result;
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
