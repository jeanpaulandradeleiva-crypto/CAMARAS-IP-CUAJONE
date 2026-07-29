// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include "cuajone/types.hpp"

#include <opencv2/core/mat.hpp>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <optional>
#include <stop_token>
#include <string>
#include <thread>

namespace cuajone {

class LatestFrameCapture {
public:
    LatestFrameCapture(
        std::string source,
        std::chrono::duration<double> reconnect_delay,
        std::chrono::duration<double> maximum_reconnect_delay,
        std::chrono::milliseconds open_timeout,
        std::chrono::milliseconds read_timeout,
        RtspTransport rtsp_transport);
    ~LatestFrameCapture();

    LatestFrameCapture(const LatestFrameCapture&) = delete;
    LatestFrameCapture& operator=(const LatestFrameCapture&) = delete;

    void start();
    void stop();
    bool waitForLatest(
        std::uint64_t previous_sequence,
        cv::Mat& frame,
        std::uint64_t& sequence,
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
    mutable std::mutex mutex_;
    std::condition_variable_any condition_;
    cv::Mat latest_;
    std::uint64_t sequence_{};
    bool ended_{};
    std::optional<std::string> error_;
    std::jthread reader_;
};

}  // namespace cuajone
