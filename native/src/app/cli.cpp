// SPDX-License-Identifier: AGPL-3.0-only

#include "cuajone/cli.hpp"
#include "cuajone/yolo_decode.hpp"

#include <algorithm>
#include <charconv>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <vector>

namespace cuajone {
namespace {

template <typename Number>
Number parseNumber(std::string_view text, std::string_view option) {
    Number value{};
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) {
        throw std::invalid_argument("Invalid value for " + std::string(option) + ": " + std::string(text));
    }
    if constexpr (std::is_floating_point_v<Number>) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(std::string(option) + " must be finite");
        }
    }
    return value;
}

std::string requireValue(int& index, int argc, char** argv, std::string_view option) {
    if (++index >= argc) {
        throw std::invalid_argument("Missing value for " + std::string(option));
    }
    return argv[index];
}

std::map<int, std::string> parseLabels(const std::string& text) {
    std::map<int, std::string> result;
    std::stringstream stream(text);
    std::string label;
    int id = 0;
    while (std::getline(stream, label, ',')) {
        if (label.empty()) throw std::invalid_argument("--ppe-labels cannot contain empty labels");
        result.emplace(id++, label);
    }
    if (result.empty()) throw std::invalid_argument("--ppe-labels requires at least one label");
    return result;
}

std::array<int, 2> parsePair(const std::string& text, std::string_view option) {
    const auto comma = text.find(',');
    if (comma == std::string::npos || text.find(',', comma + 1) != std::string::npos) {
        throw std::invalid_argument(std::string(option) + " requires two comma-separated integers");
    }
    const int first = parseNumber<int>(std::string_view(text).substr(0, comma), option);
    const int second = parseNumber<int>(std::string_view(text).substr(comma + 1), option);
    if (first <= 0 || second < 3) {
        throw std::invalid_argument(std::string(option) + " requires positive count and at least 3 dimensions");
    }
    return {first, second};
}

std::pair<std::size_t, float> parsePpeClassConfidence(
    const std::string& text,
    std::string_view option) {
    const auto separator = text.find('=');
    if (separator == std::string::npos || separator == 0 || separator + 1 == text.size()
        || text.find('=', separator + 1) != std::string::npos) {
        throw std::invalid_argument(
            std::string(option) + " requires semantic=value");
    }
    const std::string_view semantic(text.data(), separator);
    const auto found = std::ranges::find(kPpeOutputLabels, semantic);
    if (found == kPpeOutputLabels.end()) {
        throw std::invalid_argument(
            std::string(option) + " contains an unknown PPE semantic: " + std::string(semantic));
    }
    const float value = parseNumber<float>(
        std::string_view(text).substr(separator + 1), option);
    if (!std::isfinite(value) || value < 0.0F || value > 1.0F) {
        throw std::invalid_argument(std::string(option) + " must be finite and in [0, 1]");
    }
    return {static_cast<std::size_t>(found - kPpeOutputLabels.begin()), value};
}

std::chrono::milliseconds parseMilliseconds(
    const std::string& text,
    std::string_view option) {
    const std::uint64_t value = parseNumber<std::uint64_t>(text, option);
    if (value > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(std::string(option) + " exceeds OpenCV's integer millisecond range");
    }
    return std::chrono::milliseconds(value);
}

RtspTransport parseRtspTransport(std::string_view text) {
    if (text == "default") return RtspTransport::Default;
    if (text == "tcp") return RtspTransport::Tcp;
    if (text == "udp") return RtspTransport::Udp;
    throw std::invalid_argument("--rtsp-transport must be default, tcp, or udp");
}

AnalyticsMode parseAnalyticsMode(std::string_view text) {
    if (text == "ppe-only") return AnalyticsMode::PpeOnly;
    if (text == "ppe-fall") return AnalyticsMode::PpeFall;
    throw std::invalid_argument("--mode must be ppe-only or ppe-fall");
}

struct ParsedRtspSource {
    std::string host;
};

ParsedRtspSource parseRtspSource(const std::string& source) {
    if (!isRtspSource(source)) {
        throw std::invalid_argument("Source is not an rtsp:// or rtsps:// URL");
    }
    const std::size_t scheme_end = source.find("://");
    const std::size_t authority_start = scheme_end + 3;
    const std::size_t authority_end = source.find_first_of("/?#", authority_start);
    const std::size_t end = authority_end == std::string::npos ? source.size() : authority_end;
    const std::string_view authority(source.data() + authority_start, end - authority_start);
    if (authority.empty()) throw std::invalid_argument("RTSP source authority must not be empty");
    if (std::any_of(authority.begin(), authority.end(), [](unsigned char character) {
            return std::isspace(character) != 0 || character < 0x20U;
        })) {
        throw std::invalid_argument("RTSP source authority contains whitespace or control characters");
    }

    const std::size_t at = authority.rfind('@');
    const std::string_view host_and_port = at == std::string_view::npos
        ? authority : authority.substr(at + 1);
    if (at == 0) throw std::invalid_argument("RTSP source credentials must not be empty");
    if (host_and_port.empty()) throw std::invalid_argument("RTSP source host must not be empty");

    std::string_view host;
    std::string_view port;
    if (host_and_port.front() == '[') {
        const std::size_t closing = host_and_port.find(']');
        if (closing == std::string_view::npos || closing == 1) {
            throw std::invalid_argument("RTSP bracketed IPv6 host is malformed");
        }
        host = host_and_port.substr(1, closing - 1);
        const std::string_view remainder = host_and_port.substr(closing + 1);
        if (!remainder.empty()) {
            if (remainder.front() != ':') {
                throw std::invalid_argument("RTSP bracketed IPv6 host has invalid authority suffix");
            }
            port = remainder.substr(1);
        }
    } else {
        const std::size_t first_colon = host_and_port.find(':');
        const std::size_t last_colon = host_and_port.rfind(':');
        if (first_colon != std::string_view::npos && first_colon != last_colon) {
            throw std::invalid_argument("RTSP IPv6 hosts must use brackets");
        }
        if (last_colon == std::string_view::npos) host = host_and_port;
        else {
            host = host_and_port.substr(0, last_colon);
            port = host_and_port.substr(last_colon + 1);
        }
    }
    if (host.empty()) throw std::invalid_argument("RTSP source host must not be empty");
    if (host.find('@') != std::string_view::npos || host.find('[') != std::string_view::npos
        || host.find(']') != std::string_view::npos) {
        throw std::invalid_argument("RTSP source host is malformed");
    }
    if ((!port.empty() || host_and_port.ends_with(':'))) {
        if (port.empty()) throw std::invalid_argument("RTSP source port must not be empty");
        unsigned int port_number{};
        const auto converted = std::from_chars(port.data(), port.data() + port.size(), port_number);
        if (converted.ec != std::errc{} || converted.ptr != port.data() + port.size()
            || port_number == 0 || port_number > 65535) {
            throw std::invalid_argument("RTSP source port must be an integer in [1, 65535]");
        }
    }
    return {std::string(host)};
}

void validate(RuntimeConfig& config) {
    if (config.help || config.hardware_probe_json) return;
    if (config.source.empty() || config.output.empty()) {
        throw std::invalid_argument(
            "--source and --output are required");
    }
    const bool gpu_models = (!config.ppe_engine.empty()
            && (config.analytics_mode == AnalyticsMode::PpeOnly || !config.pose_engine.empty()))
        || (!config.ppe_onnx.empty()
            && (config.analytics_mode == AnalyticsMode::PpeOnly || !config.pose_onnx.empty()));
    const bool cpu_models = !config.ppe_onnx.empty()
        && (config.analytics_mode == AnalyticsMode::PpeOnly || !config.pose_onnx.empty());
    if (config.compute_backend == ComputeBackend::Cuda && !gpu_models) {
        throw std::invalid_argument(
            "CUDA mode requires PPE/pose TensorRT engines or ONNX models in ppe-fall mode");
    }
    if (config.compute_backend == ComputeBackend::Cpu && !cpu_models) {
        throw std::invalid_argument(
            "CPU mode requires --ppe-onnx and --pose-onnx in ppe-fall mode");
    }
    if (config.compute_backend == ComputeBackend::Auto && !gpu_models && !cpu_models) {
        throw std::invalid_argument(
            "Auto mode requires a complete TensorRT engine pair, ONNX model pair, or both");
    }
    validateRtspSource(config.source);
    if (config.source_label.empty()) config.source_label = defaultSourceLabel(config.source);
    const auto ratio = [](float value, std::string_view name) {
        if (!std::isfinite(value) || value < 0.0F || value > 1.0F) {
            throw std::invalid_argument(std::string(name) + " must be finite and in [0, 1]");
        }
    };
    ratio(config.ppe_confidence, "--ppe-conf");
    validateImageSize(config.image_size);
    validatePpeClassConfidences(config.ppe_class_confidences);
    ratio(config.pose_confidence, "--pose-conf");
    ratio(config.nms_iou, "--nms-iou");
    ratio(config.tracker_high_threshold, "--tracker-high-threshold");
    ratio(config.tracker_match_threshold, "--tracker-match-threshold");
    ratio(config.ppe.present_ratio, "--ppe-present-ratio");
    if ((config.device && *config.device < 0) || config.pose_class_count == 0 || config.max_det == 0
        || config.max_det > DecodeLimits{}.max_nms_candidates
        || config.tracker_high_threshold <= 0.20F
        || config.tracker_max_age == 0 || config.tracker_max_tracks == 0
        || config.tracker_frame_rate <= 0
        || config.target_fps < 0.0 || config.reconnect_delay_seconds < 0.0
        || config.maximum_reconnect_delay_seconds < config.reconnect_delay_seconds) {
        throw std::invalid_argument("Device, detection, tracker, FPS, or reconnect settings are outside supported ranges");
    }
    if (config.ppe.window == 0 || config.ppe.minimum_samples == 0
        || config.ppe.minimum_samples > config.ppe.window) {
        throw std::invalid_argument("--ppe-min-samples must be in [1, --ppe-window]");
    }
    const auto nonnegativeDuration = [](std::chrono::duration<double> value, std::string_view name) {
        if (!std::isfinite(value.count()) || value.count() < 0.0) {
            throw std::invalid_argument(std::string(name) + " must be finite and non-negative");
        }
    };
    nonnegativeDuration(config.ppe.alert_cooldown, "--ppe-cooldown");
    nonnegativeDuration(config.ppe.track_ttl, "--ppe-track-ttl");
    nonnegativeDuration(config.fall.alert_cooldown, "--fall-cooldown");
    nonnegativeDuration(config.fall.track_ttl, "--fall-track-ttl");
    if (config.fall.confirm_frames == 0 || config.fall.reset_frames == 0) {
        throw std::invalid_argument("Fall confirmation and reset frame counts must be positive");
    }
    if (!std::isfinite(config.fall.aspect_ratio) || config.fall.aspect_ratio <= 0.0F
        || !std::isfinite(config.fall.torso_angle_degrees)
        || config.fall.torso_angle_degrees < 0.0F || config.fall.torso_angle_degrees > 90.0F
        || !std::isfinite(config.fall.descent_ratio) || config.fall.descent_ratio < 0.0F
        || !std::isfinite(config.fall.near_floor_ratio)
        || config.fall.near_floor_ratio < 0.0F || config.fall.near_floor_ratio > 1.0F) {
        throw std::invalid_argument("Fall geometry thresholds are outside supported finite ranges");
    }
}

}  // namespace

RuntimeConfig parseCommandLine(int argc, char** argv) {
    RuntimeConfig config;
    std::array<std::optional<float>, kPpeOutputLabels.size()> class_confidence_overrides;
    for (int index = 1; index < argc; ++index) {
        const std::string_view option = argv[index];
        if (option == "--help" || option == "-h") config.help = true;
        else if (option == "--preflight") config.preflight = true;
        else if (option == "--hardware-probe-json") config.hardware_probe_json = true;
        else if (option == "--show") config.show_window = true;
        else if (option == "--allow-nonperson-pose-class") config.allow_nonperson_pose_class = true;
        else if (option == "--mode") config.analytics_mode = parseAnalyticsMode(requireValue(index, argc, argv, option));
        else if (option == "--source") config.source = requireValue(index, argc, argv, option);
        else if (option == "--source-label") config.source_label = requireValue(index, argc, argv, option);
        else if (option == "--compute") {
            config.compute_backend = parseComputeBackend(requireValue(index, argc, argv, option));
            config.compute_explicit = true;
        }
        else if (option == "--ppe-engine") config.ppe_engine = requireValue(index, argc, argv, option);
        else if (option == "--pose-engine") config.pose_engine = requireValue(index, argc, argv, option);
        else if (option == "--ppe-onnx") config.ppe_onnx = requireValue(index, argc, argv, option);
        else if (option == "--pose-onnx") config.pose_onnx = requireValue(index, argc, argv, option);
        else if (option == "--output") config.output = requireValue(index, argc, argv, option);
        else if (option == "--ppe-labels") config.ppe_labels = parseLabels(requireValue(index, argc, argv, option));
        else if (option == "--pose-class-count") config.pose_class_count = parseNumber<std::size_t>(requireValue(index, argc, argv, option), option);
        else if (option == "--pose-kpt-shape") config.pose_keypoint_shape = parsePair(requireValue(index, argc, argv, option), option);
        else if (option == "--device") config.device = parseNumber<int>(requireValue(index, argc, argv, option), option);
        else if (option == "--imgsz") config.image_size = parseNumber<int>(requireValue(index, argc, argv, option), option);
        else if (option == "--ppe-conf") config.ppe_confidence = parseNumber<float>(requireValue(index, argc, argv, option), option);
        else if (option == "--ppe-class-conf") {
            const auto [class_index, value] = parsePpeClassConfidence(
                requireValue(index, argc, argv, option), option);
            if (class_confidence_overrides[class_index]) {
                throw std::invalid_argument(
                    "Duplicate --ppe-class-conf semantic: "
                    + std::string(kPpeOutputLabels[class_index]));
            }
            class_confidence_overrides[class_index] = value;
        }
        else if (option == "--pose-conf") config.pose_confidence = parseNumber<float>(requireValue(index, argc, argv, option), option);
        else if (option == "--nms-iou") config.nms_iou = parseNumber<float>(requireValue(index, argc, argv, option), option);
        else if (option == "--max-det") config.max_det = parseNumber<std::size_t>(requireValue(index, argc, argv, option), option);
        else if (option == "--tracker-high-threshold") config.tracker_high_threshold = parseNumber<float>(requireValue(index, argc, argv, option), option);
        else if (option == "--tracker-match-threshold") config.tracker_match_threshold = parseNumber<float>(requireValue(index, argc, argv, option), option);
        else if (option == "--tracker-iou") config.tracker_match_threshold = 1.0F - parseNumber<float>(requireValue(index, argc, argv, option), option);
        else if (option == "--tracker-max-age") config.tracker_max_age = parseNumber<std::size_t>(requireValue(index, argc, argv, option), option);
        else if (option == "--tracker-max-tracks") config.tracker_max_tracks = parseNumber<std::size_t>(requireValue(index, argc, argv, option), option);
        else if (option == "--tracker-frame-rate") config.tracker_frame_rate = parseNumber<int>(requireValue(index, argc, argv, option), option);
        else if (option == "--target-fps") config.target_fps = parseNumber<double>(requireValue(index, argc, argv, option), option);
        else if (option == "--reconnect-delay") config.reconnect_delay_seconds = parseNumber<double>(requireValue(index, argc, argv, option), option);
        else if (option == "--max-reconnect-delay") config.maximum_reconnect_delay_seconds = parseNumber<double>(requireValue(index, argc, argv, option), option);
        else if (option == "--capture-open-timeout-ms") config.capture_open_timeout = parseMilliseconds(requireValue(index, argc, argv, option), option);
        else if (option == "--capture-read-timeout-ms") config.capture_read_timeout = parseMilliseconds(requireValue(index, argc, argv, option), option);
        else if (option == "--rtsp-transport") config.rtsp_transport = parseRtspTransport(requireValue(index, argc, argv, option));
        else if (option == "--ppe-window") config.ppe.window = parseNumber<std::size_t>(requireValue(index, argc, argv, option), option);
        else if (option == "--ppe-min-samples") config.ppe.minimum_samples = parseNumber<std::size_t>(requireValue(index, argc, argv, option), option);
        else if (option == "--ppe-present-ratio") config.ppe.present_ratio = parseNumber<float>(requireValue(index, argc, argv, option), option);
        else if (option == "--ppe-cooldown") config.ppe.alert_cooldown = std::chrono::duration<double>(parseNumber<double>(requireValue(index, argc, argv, option), option));
        else if (option == "--ppe-track-ttl") config.ppe.track_ttl = std::chrono::duration<double>(parseNumber<double>(requireValue(index, argc, argv, option), option));
        else if (option == "--fall-confirm-frames") config.fall.confirm_frames = parseNumber<std::size_t>(requireValue(index, argc, argv, option), option);
        else if (option == "--fall-reset-frames") config.fall.reset_frames = parseNumber<std::size_t>(requireValue(index, argc, argv, option), option);
        else if (option == "--fall-cooldown") config.fall.alert_cooldown = std::chrono::duration<double>(parseNumber<double>(requireValue(index, argc, argv, option), option));
        else if (option == "--fall-track-ttl") config.fall.track_ttl = std::chrono::duration<double>(parseNumber<double>(requireValue(index, argc, argv, option), option));
        else if (option == "--fall-aspect-ratio") config.fall.aspect_ratio = parseNumber<float>(requireValue(index, argc, argv, option), option);
        else if (option == "--fall-torso-angle") config.fall.torso_angle_degrees = parseNumber<float>(requireValue(index, argc, argv, option), option);
        else if (option == "--fall-descent-ratio") config.fall.descent_ratio = parseNumber<float>(requireValue(index, argc, argv, option), option);
        else if (option == "--fall-near-floor-ratio") config.fall.near_floor_ratio = parseNumber<float>(requireValue(index, argc, argv, option), option);
        else throw std::invalid_argument("Unknown option: " + std::string(option));
    }
    config.ppe_class_confidences.fill(config.ppe_confidence);
    for (std::size_t index = 0; index < class_confidence_overrides.size(); ++index) {
        if (class_confidence_overrides[index]) {
            config.ppe_class_confidences[index] = *class_confidence_overrides[index];
        }
    }
    validate(config);
    return config;
}

void printHelp(std::ostream& output) {
    output <<
        "NexoAI Vision PPE and fall analytics\n\n"
        "Required:\n"
        "  --source <rtsp-or-file>       RTSP URL, video, or image\n"
        "  --output <directory>         Evidence and append-only CSV directory\n\n"
        "Compute and models:\n"
        "  --compute <mode>             auto, cuda, or cpu (default: installed setting/auto)\n"
        "  --ppe-engine <file.engine>   PPE TensorRT engine for CUDA\n"
        "  --pose-engine <file.engine>  Pose TensorRT engine for CUDA ppe-fall\n"
        "  --ppe-onnx <file.onnx>       PPE ONNX model for CPU or CUDA\n"
        "  --pose-onnx <file.onnx>      Pose ONNX model for CPU or hybrid ppe-fall (CPU session)\n\n"
        "Diagnostics and identity:\n"
        "  --help                       Show this help without runtime startup\n"
        "  --hardware-probe-json        Print stable NVIDIA/CUDA probe JSON and exit\n"
        "  --preflight                  Validate everything without opening the source\n"
        "  --mode <mode>                ppe-only or ppe-fall (default: ppe-fall)\n"
        "  --device <index>             CUDA device index (default: first compatible)\n"
        "  --source-label <label>       Non-secret source label used in events\n"
        "  --ppe-labels <a,b,c>         Class labels for a raw engine without metadata\n"
        "  --pose-class-count <number>  Pose classes fallback (default: 1)\n"
        "  --pose-kpt-shape <count,dim> Pose schema fallback (default: 17,3)\n"
        "  --allow-nonperson-pose-class Allow non-person single-class pose metadata\n\n"
        "Core thresholds:\n"
        "  --imgsz <size>               Inference size: 640, 768, 960, or 1280 (default: 640)\n"
        "  --ppe-conf <0..1>            PPE confidence (default: 0.30)\n"
        "  --ppe-class-conf <name=value> Repeatable exact PPE class threshold override\n"
        "  --pose-conf <0..1>           Pose confidence (default: 0.35)\n"
        "  --nms-iou <0..1>             Class-aware NMS IoU (default: 0.45)\n"
        "  --max-det <number>           Final detections per engine (default: 300)\n"
        "  --tracker-high-threshold <0..1> ByteTrack new-track confidence (default: 0.35)\n"
        "  --tracker-match-threshold <0..1> ByteTrack assignment cost limit (default: 0.80)\n"
        "  --tracker-iou <0..1>         Legacy minimum-IoU alias converted to match cost\n"
        "  --tracker-max-age <number>   Positive ByteTrack lost-frame buffer (default: 30)\n"
        "  --tracker-max-tracks <count> Positive retained-track capacity (default: 128)\n"
        "  --tracker-frame-rate <fps>   Positive ByteTrack reference FPS (default: 30)\n"
        "  --target-fps <number>        Non-negative; 0 processes every latest frame\n"
        "  --show                       Display the annotated OpenCV window\n\n"
        "Capture and RTSP:\n"
        "  --rtsp-transport <mode>      default, tcp, or udp (default: default)\n"
        "  --reconnect-delay <seconds>  Non-negative initial delay (default: 5)\n"
        "  --max-reconnect-delay <sec>  At least reconnect delay (default: 30)\n"
        "  --capture-open-timeout-ms <n> Non-negative FFmpeg open timeout (default: 20000)\n"
        "  --capture-read-timeout-ms <n> Non-negative FFmpeg read timeout (default: 10000)\n\n"
        "PPE voting:\n"
        "  --ppe-window <number>         Positive voting capacity (default: 20)\n"
        "  --ppe-min-samples <number>    In [1, window] (default: 12)\n"
        "  --ppe-present-ratio <0..1>    Required presence ratio (default: 0.35)\n"
        "  --ppe-cooldown <seconds>      Non-negative (default: 60)\n"
        "  --ppe-track-ttl <seconds>     Non-negative (default: 5)\n\n"
        "Fall analytics:\n"
        "  --fall-confirm-frames <n>    Positive confirmation count (default: 12)\n"
        "  --fall-reset-frames <n>      Positive reset count (default: 20)\n"
        "  --fall-cooldown <seconds>    Non-negative (default: 120)\n"
        "  --fall-track-ttl <seconds>   Non-negative (default: 5)\n"
        "  --fall-aspect-ratio <value>  Positive finite threshold (default: 1.05)\n"
        "  --fall-torso-angle <degrees> Finite threshold in [0,90] (default: 55)\n"
        "  --fall-descent-ratio <value> Non-negative finite threshold (default: 0.12)\n"
        "  --fall-near-floor-ratio <0..1> Finite threshold (default: 0.65)\n\n"
        "RTSP URLs require a non-empty host; bracket IPv6 addresses. Credentials are redacted.\n";
}

std::string redactSource(const std::string& source) {
    const auto scheme = source.find("://");
    if (scheme == std::string::npos) return source;
    const auto authority_start = scheme + 3;
    const auto authority_end = source.find_first_of("/?#", authority_start);
    const std::size_t end = authority_end == std::string::npos ? source.size() : authority_end;
    const std::string_view authority(source.data() + authority_start, end - authority_start);
    const auto relative_at = authority.rfind('@');
    if (relative_at == std::string_view::npos) return source;
    const std::size_t at = authority_start + relative_at;
    return source.substr(0, authority_start) + "***@" + source.substr(at + 1);
}

std::string defaultSourceLabel(const std::string& source) {
    if (isRtspSource(source)) {
        return "rtsp-" + parseRtspSource(source).host;
    }
    const std::filesystem::path path(source);
    return path.filename().string().empty() ? "offline-source" : path.filename().string();
}

bool isRtspSource(std::string_view source) noexcept {
    return source.starts_with("rtsp://") || source.starts_with("rtsps://");
}

void validateRtspSource(const std::string& source) {
    if (isRtspSource(source)) static_cast<void>(parseRtspSource(source));
}

}  // namespace cuajone
