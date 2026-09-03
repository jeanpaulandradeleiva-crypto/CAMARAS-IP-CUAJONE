// SPDX-License-Identifier: AGPL-3.0-only

#include "cuajone/capture.hpp"
#include "cuajone/cli.hpp"
#include "cuajone/performance_telemetry.hpp"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#endif

#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <thread>
#include <vector>

namespace cuajone {
namespace {

bool isLiteralIpv4(std::string_view host) noexcept {
    std::size_t component_start{};
    for (std::size_t component_index = 0; component_index < 4; ++component_index) {
        const std::size_t component_end = host.find('.', component_start);
        if (component_index < 3 && component_end == std::string_view::npos) return false;
        const std::string_view component = host.substr(
            component_start,
            component_end == std::string_view::npos
                ? std::string_view::npos : component_end - component_start);
        if (component.empty() || component.size() > 3) return false;
        unsigned int value{};
        for (const char character : component) {
            if (character < '0' || character > '9') return false;
            value = value * 10U + static_cast<unsigned int>(character - '0');
        }
        if (value > 255U) return false;
        if (component_index == 3) return component_end == std::string_view::npos;
        component_start = component_end + 1;
    }
    return false;
}

bool countIpv6Groups(std::string_view groups, std::size_t& count) noexcept {
    while (!groups.empty()) {
        const std::size_t separator = groups.find(':');
        const std::string_view group = groups.substr(0, separator);
        if (group.empty()) return false;
        if (group.find('.') != std::string_view::npos) {
            if (separator != std::string_view::npos || !isLiteralIpv4(group)) return false;
            count += 2;
            return true;
        }
        if (group.size() > 4) return false;
        for (const char character : group) {
            if (std::isxdigit(static_cast<unsigned char>(character)) == 0) return false;
        }
        ++count;
        if (separator == std::string_view::npos) return true;
        groups.remove_prefix(separator + 1);
    }
    return true;
}

bool isLiteralIpv6(std::string_view host) noexcept {
    if (host.empty() || host.find(':') == std::string_view::npos || host.find(":::") != std::string_view::npos) {
        return false;
    }
    const std::size_t compression = host.find("::");
    if (compression == std::string_view::npos) {
        std::size_t count{};
        return countIpv6Groups(host, count) && count == 8;
    }
    if (host.find("::", compression + 2) != std::string_view::npos) return false;
    std::size_t count{};
    return countIpv6Groups(host.substr(0, compression), count)
        && countIpv6Groups(host.substr(compression + 2), count) && count < 8;
}

#ifdef _WIN32
class WinsockSession {
public:
    WinsockSession() : started_(WSAStartup(MAKEWORD(2, 2), &data_) == 0) {}
    ~WinsockSession() {
        if (started_) WSACleanup();
    }

    [[nodiscard]] bool started() const noexcept { return started_; }

private:
    WSADATA data_{};
    bool started_{};
};

int reasonPriority(RtspReachabilityReason reason) {
    switch (reason) {
        case RtspReachabilityReason::NoRoute: return 4;
        case RtspReachabilityReason::TcpTimeout: return 3;
        case RtspReachabilityReason::ConnectionRefused: return 2;
        case RtspReachabilityReason::Unknown: return 1;
        default: return 0;
    }
}

void retainMoreSpecificReason(
    std::optional<RtspReachabilityReason>& retained,
    RtspReachabilityReason candidate) {
    if (!retained || reasonPriority(candidate) > reasonPriority(*retained)) retained = candidate;
}
#endif

RtspReachabilityReason probeRtspReachability(
    const RtspAuthority& authority,
    std::chrono::milliseconds timeout) {
#ifdef _WIN32
    if (timeout.count() <= 0) return RtspReachabilityReason::Unknown;

    WinsockSession winsock;
    if (!winsock.started()) return RtspReachabilityReason::Unknown;

    const auto deadline = std::chrono::steady_clock::now() + timeout;
    ADDRINFOEXA hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    const RtspAddressPath address_path = classifyRtspAddressPath(authority.host);
    hints.ai_flags = AI_NUMERICSERV;
    if (address_path == RtspAddressPath::DirectAddress) hints.ai_flags |= AI_NUMERICHOST;
    PADDRINFOEXA addresses{};
    const std::string service = std::to_string(authority.port);
    timeval resolution_timeout{
        static_cast<long>(timeout.count() / 1000),
        static_cast<long>((timeout.count() % 1000) * 1000),
    };
    const int resolution_error = GetAddrInfoExA(authority.host.c_str(), service.c_str(), NS_ALL, nullptr,
        &hints, &addresses, &resolution_timeout, nullptr, nullptr, nullptr);
    if (resolution_error != 0) {
        return classifyRtspResolutionFailure(address_path, resolution_error);
    }

    std::optional<RtspReachabilityReason> failure;
    for (auto* address = addresses; address != nullptr; address = address->ai_next) {
        const SOCKET socket_handle = socket(
            address->ai_family, address->ai_socktype, address->ai_protocol);
        if (socket_handle == INVALID_SOCKET) {
            retainMoreSpecificReason(failure, classifyRtspConnectError(WSAGetLastError()));
            continue;
        }
        const auto close_socket = [&] { closesocket(socket_handle); };
        u_long non_blocking = 1;
        if (ioctlsocket(socket_handle, FIONBIO, &non_blocking) != 0) {
            retainMoreSpecificReason(failure, classifyRtspConnectError(WSAGetLastError()));
            close_socket();
            continue;
        }
        if (connect(socket_handle, address->ai_addr, static_cast<int>(address->ai_addrlen)) == 0) {
            close_socket();
            FreeAddrInfoExA(addresses);
            return RtspReachabilityReason::RtspHandshakeFailed;
        }

        const int connect_error = WSAGetLastError();
        if (connect_error != WSAEWOULDBLOCK && connect_error != WSAEINPROGRESS) {
            retainMoreSpecificReason(failure, classifyRtspConnectError(connect_error));
            close_socket();
            continue;
        }
        const auto now = std::chrono::steady_clock::now();
        if (now >= deadline) {
            retainMoreSpecificReason(failure, RtspReachabilityReason::TcpTimeout);
            close_socket();
            continue;
        }
        const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
        timeval connect_timeout{
            static_cast<long>(remaining.count() / 1000),
            static_cast<long>((remaining.count() % 1000) * 1000),
        };
        fd_set writable;
        fd_set errors;
        FD_ZERO(&writable);
        FD_ZERO(&errors);
        FD_SET(socket_handle, &writable);
        FD_SET(socket_handle, &errors);
        const int selected = select(0, nullptr, &writable, &errors, &connect_timeout);
        if (selected == 0) {
            retainMoreSpecificReason(failure, RtspReachabilityReason::TcpTimeout);
            close_socket();
            continue;
        }
        if (selected == SOCKET_ERROR) {
            retainMoreSpecificReason(failure, classifyRtspConnectError(WSAGetLastError()));
            close_socket();
            continue;
        }
        int socket_error{};
        int socket_error_size = sizeof(socket_error);
        if (getsockopt(socket_handle, SOL_SOCKET, SO_ERROR,
                reinterpret_cast<char*>(&socket_error), &socket_error_size) != 0) {
            retainMoreSpecificReason(failure, classifyRtspConnectError(WSAGetLastError()));
        } else if (socket_error == 0) {
            close_socket();
            FreeAddrInfoExA(addresses);
            return RtspReachabilityReason::RtspHandshakeFailed;
        } else {
            retainMoreSpecificReason(failure, classifyRtspConnectError(socket_error));
        }
        close_socket();
    }
    FreeAddrInfoExA(addresses);
    return failure.value_or(RtspReachabilityReason::Unknown);
#else
    (void)authority;
    (void)timeout;
    return RtspReachabilityReason::Unknown;
#endif
}

std::string safeEndpoint(std::string_view host, std::uint16_t port) {
    if (host.empty()) return "unavailable";
    const bool is_ipv6 = host.find(':') != std::string_view::npos;
    return (is_ipv6 ? "[" : "") + std::string(host) + (is_ipv6 ? "]" : "")
        + ":" + std::to_string(port);
}

}  // namespace

RtspReachabilityReason classifyRtspConnectError(int native_error) noexcept {
#ifdef _WIN32
    if (native_error == WSAENETUNREACH || native_error == WSAEHOSTUNREACH
        || native_error == WSAEHOSTDOWN) {
        return RtspReachabilityReason::NoRoute;
    }
    if (native_error == WSAETIMEDOUT) return RtspReachabilityReason::TcpTimeout;
    if (native_error == WSAECONNREFUSED) return RtspReachabilityReason::ConnectionRefused;
#else
    (void)native_error;
#endif
    return RtspReachabilityReason::Unknown;
}

RtspReachabilityReason classifyRtspResolutionFailure(
    RtspAddressPath address_path,
    int native_error) noexcept {
    if (address_path == RtspAddressPath::DirectAddress) return RtspReachabilityReason::Unknown;
#ifdef _WIN32
    if (native_error == EAI_AGAIN || native_error == EAI_FAIL || native_error == EAI_NONAME
        || native_error == EAI_NODATA) {
        return RtspReachabilityReason::DnsFailure;
    }
#else
    (void)native_error;
#endif
    return RtspReachabilityReason::Unknown;
}

RtspAddressPath classifyRtspAddressPath(std::string_view host) noexcept {
    return isLiteralIpv4(host) || isLiteralIpv6(host)
        ? RtspAddressPath::DirectAddress
        : RtspAddressPath::HostnameResolution;
}

bool shouldSkipRtspOpen(RtspReachabilityReason reason) noexcept {
    switch (reason) {
        case RtspReachabilityReason::DnsFailure:
        case RtspReachabilityReason::NoRoute:
        case RtspReachabilityReason::TcpTimeout:
        case RtspReachabilityReason::ConnectionRefused:
            return true;
        case RtspReachabilityReason::RtspHandshakeFailed:
        case RtspReachabilityReason::Unknown:
            return false;
    }
    return false;
}

std::string formatRtspOpenProgress(std::string_view host, std::uint16_t port) {
    return "RTSP open [endpoint=" + safeEndpoint(host, port)
        + "]: checking reachability before opening RTSP.";
}

std::string formatRtspOpenFailure(
    std::string_view host,
    std::uint16_t port,
    RtspReachabilityReason reason) {
    const std::string prefix = "RTSP open failed [reason=";
    const std::string endpoint = "] endpoint=" + safeEndpoint(host, port) + ": ";
    switch (reason) {
        case RtspReachabilityReason::DnsFailure:
            return prefix + "dns_failure" + endpoint + "DNS resolution failed.";
        case RtspReachabilityReason::NoRoute:
            return prefix + "no_route" + endpoint
                + "No route to the endpoint; it is unreachable from the current network path and may be affected by routing, VLAN, firewall, or host availability.";
        case RtspReachabilityReason::TcpTimeout:
            return prefix + "tcp_timeout" + endpoint
                + "TCP connection timed out; the endpoint is unreachable from the current network path and may be affected by routing, VLAN, firewall, or host availability.";
        case RtspReachabilityReason::ConnectionRefused:
            return prefix + "connection_refused" + endpoint + "TCP connection was refused.";
        case RtspReachabilityReason::RtspHandshakeFailed:
            return prefix + "rtsp_handshake_failed" + endpoint
                + "TCP connectivity succeeded, but OpenCV/FFmpeg could not complete the RTSP handshake.";
        case RtspReachabilityReason::Unknown:
            return prefix + "unknown" + endpoint
                + "OpenCV could not open the source; reachability could not be classified.";
    }
    return prefix + "unknown" + endpoint
        + "OpenCV could not open the source; reachability could not be classified.";
}

LatestFrameCapture::LatestFrameCapture(
    std::string source,
    std::chrono::duration<double> reconnect_delay,
    std::chrono::duration<double> maximum_reconnect_delay,
    std::chrono::milliseconds open_timeout,
    std::chrono::milliseconds read_timeout,
    RtspTransport rtsp_transport,
    PerformanceTelemetry* telemetry)
    : source_(std::move(source)),
      reconnect_delay_(reconnect_delay),
       maximum_reconnect_delay_(maximum_reconnect_delay),
       open_timeout_(open_timeout),
       read_timeout_(read_timeout),
       rtsp_transport_(rtsp_transport),
       telemetry_(telemetry) {
    if (source_.empty()) throw std::invalid_argument("Capture source must not be empty");
    if (!std::isfinite(reconnect_delay_.count()) || reconnect_delay_.count() < 0.0
        || !std::isfinite(maximum_reconnect_delay_.count())
        || maximum_reconnect_delay_ < reconnect_delay_) {
        throw std::invalid_argument("Capture reconnect delays must be finite, non-negative, and ordered");
    }
    if (open_timeout_.count() < 0 || read_timeout_.count() < 0
        || open_timeout_.count() > std::numeric_limits<int>::max()
        || read_timeout_.count() > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("Capture timeouts must fit OpenCV's non-negative integer millisecond range");
    }
}

LatestFrameCapture::~LatestFrameCapture() {
    stop();
}

void LatestFrameCapture::start() {
    if (reader_.joinable()) throw std::logic_error("Capture is already running");
    {
        std::scoped_lock lock(mutex_);
        latest_.release();
        sequence_ = 0;
        published_at_ = {};
        ended_ = false;
        error_.reset();
    }
    reader_ = std::jthread([this](std::stop_token token) {
        try {
            readerLoop(token);
        } catch (const std::exception& error) {
            {
                std::scoped_lock lock(mutex_);
                error_ = error.what();
                ended_ = true;
            }
            condition_.notify_all();
        }
    });
}

void LatestFrameCapture::stop() {
    if (!reader_.joinable()) return;
    reader_.request_stop();
    condition_.notify_all();
    reader_.join();
}

bool LatestFrameCapture::waitForLatest(
    std::uint64_t previous_sequence,
    cv::Mat& frame,
    std::uint64_t& sequence,
    std::chrono::steady_clock::time_point& published_at,
    std::chrono::milliseconds timeout) {
    if (timeout.count() < 0) throw std::invalid_argument("Frame wait timeout must be non-negative");
    const auto wait_started = telemetry_ == nullptr
        ? std::chrono::steady_clock::time_point{} : std::chrono::steady_clock::now();
    std::unique_lock lock(mutex_);
    condition_.wait_for(lock, timeout, [&] {
        return sequence_ != previous_sequence || ended_;
    });
    if (telemetry_ != nullptr) {
        telemetry_->addSample(
            PerformanceStage::CaptureWait, std::chrono::steady_clock::now() - wait_started);
        // A wait timeout only means no newer frame arrived in time; it is not a
        // frame drop and must stay distinct from latest_slot_sequence_gap_drops.
        if (sequence_ == previous_sequence && !ended_) telemetry_->recordCaptureWaitTimeout();
    }
    if (sequence_ == previous_sequence) return false;
    frame = latest_;
    sequence = sequence_;
    if (telemetry_ != nullptr) published_at = published_at_;
    return true;
}

bool LatestFrameCapture::ended() const {
    std::scoped_lock lock(mutex_);
    return ended_;
}

std::optional<std::string> LatestFrameCapture::lastError() const {
    std::scoped_lock lock(mutex_);
    return error_;
}

bool LatestFrameCapture::isRtsp() const {
    return source_.rfind("rtsp://", 0) == 0 || source_.rfind("rtsps://", 0) == 0;
}

bool LatestFrameCapture::isImage() const {
    std::string extension = std::filesystem::path(source_).extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return extension == ".jpg" || extension == ".jpeg" || extension == ".png" || extension == ".bmp";
}

void LatestFrameCapture::publish(cv::Mat frame) {
    {
        std::scoped_lock lock(mutex_);
        latest_ = std::move(frame);
        ++sequence_;
        if (telemetry_ != nullptr) published_at_ = std::chrono::steady_clock::now();
        error_.reset();
    }
    if (telemetry_ != nullptr) telemetry_->capturedFrame();
    condition_.notify_all();
}

void LatestFrameCapture::readerLoop(std::stop_token stop_token) {
    if (isImage()) {
        cv::Mat image = cv::imread(source_, cv::IMREAD_COLOR);
        if (image.empty()) {
            std::scoped_lock lock(mutex_);
            error_ = "OpenCV could not read the image source";
        } else {
            publish(std::move(image));
        }
        {
            std::scoped_lock lock(mutex_);
            ended_ = true;
        }
        condition_.notify_all();
        return;
    }

    if (isRtsp() && rtsp_transport_ != RtspTransport::Default) {
        const char* options = rtsp_transport_ == RtspTransport::Tcp
            ? "rtsp_transport;tcp" : "rtsp_transport;udp";
#ifdef _WIN32
        if (_putenv_s("OPENCV_FFMPEG_CAPTURE_OPTIONS", options) != 0) {
            throw std::runtime_error("Could not configure OpenCV FFmpeg RTSP transport");
        }
#else
        if (setenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", options, 1) != 0) {
            throw std::runtime_error("Could not configure OpenCV FFmpeg RTSP transport");
        }
#endif
    }

    auto delay = reconnect_delay_;
    while (!stop_token.stop_requested()) {
        cv::VideoCapture capture;
        const bool rtsp = isRtsp();
        std::string host;
        std::uint16_t port{};
        RtspReachabilityReason preflight_reason = RtspReachabilityReason::Unknown;
        bool tcp_connectivity_established = false;
        bool skip_open = false;
        if (rtsp) {
            try {
                const RtspAuthority authority = parseRtspAuthority(source_);
                host = authority.host;
                port = authority.port;
                std::cerr << formatRtspOpenProgress(host, port) << '\n';
                preflight_reason = probeRtspReachability(authority, kRtspPreflightTimeout);
                tcp_connectivity_established =
                    preflight_reason == RtspReachabilityReason::RtspHandshakeFailed;
                skip_open = shouldSkipRtspOpen(preflight_reason);
            } catch (const std::exception&) {
                // Keep the credential-safe unknown fallback for malformed direct construction.
                std::cerr << formatRtspOpenProgress(host, port) << '\n';
            }
        }

        bool opened = false;
        if (!skip_open) {
            std::vector<int> parameters;
            if (open_timeout_.count() > 0) {
                parameters.insert(parameters.end(), {
                    cv::CAP_PROP_OPEN_TIMEOUT_MSEC, static_cast<int>(open_timeout_.count())});
            }
            if (read_timeout_.count() > 0) {
                parameters.insert(parameters.end(), {
                    cv::CAP_PROP_READ_TIMEOUT_MSEC, static_cast<int>(read_timeout_.count())});
            }
            opened = parameters.empty()
                ? capture.open(source_, cv::CAP_FFMPEG)
                : capture.open(source_, cv::CAP_FFMPEG, parameters);
            if (!opened && !parameters.empty()) {
                capture.release();
                opened = capture.open(source_, cv::CAP_FFMPEG);
            }
        }
        if (!opened) {
            std::string open_error = "OpenCV could not open the source";
            if (rtsp) {
                const RtspReachabilityReason reason = skip_open
                    ? preflight_reason
                    : tcp_connectivity_established
                        ? RtspReachabilityReason::RtspHandshakeFailed
                        : RtspReachabilityReason::Unknown;
                open_error = formatRtspOpenFailure(host, port, reason);
            }
            std::cerr << open_error << '\n';
            {
                std::scoped_lock lock(mutex_);
                error_ = std::move(open_error);
            }
            condition_.notify_all();
        } else {
            capture.set(cv::CAP_PROP_BUFFERSIZE, 1);
            delay = reconnect_delay_;
            const double source_fps = isRtsp() ? 0.0 : capture.get(cv::CAP_PROP_FPS);
            const bool pace_offline_video = source_fps > 0.0 && source_fps <= 240.0;
            const auto frame_period = pace_offline_video
                ? std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                    std::chrono::duration<double>(1.0 / source_fps))
                : std::chrono::steady_clock::duration::zero();
            auto next_frame_at = std::chrono::steady_clock::now();
            while (!stop_token.stop_requested()) {
                cv::Mat next;
                if (!capture.read(next) || next.empty()) break;
                publish(std::move(next));
                if (pace_offline_video) {
                    next_frame_at += frame_period;
                    while (!stop_token.stop_requested()
                           && std::chrono::steady_clock::now() < next_frame_at) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(2));
                    }
                }
            }
            capture.release();
            if (!isRtsp()) break;
            {
                std::scoped_lock lock(mutex_);
                error_ = "RTSP stream disconnected; reconnect is pending";
            }
            condition_.notify_all();
        }

        if (!isRtsp() || stop_token.stop_requested()) break;
        const auto deadline = std::chrono::steady_clock::now() + delay;
        std::unique_lock lock(mutex_);
        condition_.wait_until(lock, stop_token, deadline, [] { return false; });
        lock.unlock();
        delay = std::min(delay * 2.0, maximum_reconnect_delay_);
    }
    {
        std::scoped_lock lock(mutex_);
        ended_ = true;
    }
    condition_.notify_all();
}

}  // namespace cuajone
