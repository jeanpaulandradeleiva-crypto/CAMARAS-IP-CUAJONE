// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include "cuajone/types.hpp"

namespace cuajone { class PerformanceTelemetry; }

#include <opencv2/core/mat.hpp>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <optional>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>

namespace cuajone {

enum class RtspReachabilityReason {
    DnsFailure,
    NoRoute,
    TcpTimeout,
    ConnectionRefused,
    RtspHandshakeFailed,
    Unknown,
};

enum class RtspAddressPath {
    DirectAddress,
    HostnameResolution,
};

// Fixed, independent bound for the Windows DNS/TCP preflight before RTSP open.
inline constexpr std::chrono::milliseconds kRtspPreflightTimeout{2000};

[[nodiscard]] RtspReachabilityReason classifyRtspConnectError(int native_error) noexcept;
[[nodiscard]] RtspAddressPath classifyRtspAddressPath(std::string_view host) noexcept;
[[nodiscard]] RtspReachabilityReason classifyRtspResolutionFailure(
    RtspAddressPath address_path,
    int native_error) noexcept;
[[nodiscard]] bool shouldSkipRtspOpen(RtspReachabilityReason reason) noexcept;
[[nodiscard]] std::string formatRtspOpenProgress(std::string_view host, std::uint16_t port);
[[nodiscard]] std::string formatRtspOpenFailure(
    std::string_view host,
    std::uint16_t port,
    RtspReachabilityReason reason);

class LatestFrameCapture {
public:
    LatestFrameCapture(
        std::string source,
        std::chrono::duration<double> reconnect_delay,
        std::chrono::duration<double> maximum_reconnect_delay,
        std::chrono::milliseconds open_timeout,
        std::chrono::milliseconds read_timeout,
        RtspTransport rtsp_transport,
        PerformanceTelemetry* telemetry = nullptr);
    ~LatestFrameCapture();

    LatestFrameCapture(const LatestFrameCapture&) = delete;
    LatestFrameCapture& operator=(const LatestFrameCapture&) = delete;

    void start();
    void stop();
    bool waitForLatest(
        std::uint64_t previous_sequence,
        cv::Mat& frame,
        std::uint64_t& sequence,
        std::chrono::steady_clock::time_point& published_at,
        std::chrono::milliseconds timeout);
    [[nodiscard]] bool ended() const;
    [[nodiscard]] std::optional<std::string> lastError() const;

private:
    void readerLoop(std::stop_token stop_token);
    bool isRtsp() const;
    bool isImage() const;
    void publish(cv::Mat frame);

    std::string source_;
    std::chrono::duration<double> reconnect_delay_;
    std::chrono::duration<double> maximum_reconnect_delay_;
    std::chrono::milliseconds open_timeout_;
    std::chrono::milliseconds read_timeout_;
    RtspTransport rtsp_transport_;
    PerformanceTelemetry* telemetry_{};
    mutable std::mutex mutex_;
    std::condition_variable_any condition_;
    cv::Mat latest_;
    std::uint64_t sequence_{};
    std::chrono::steady_clock::time_point published_at_{};
    bool ended_{};
    std::optional<std::string> error_;
    std::jthread reader_;
};

}  // namespace cuajone
