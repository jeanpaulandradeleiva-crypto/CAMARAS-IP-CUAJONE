// SPDX-License-Identifier: AGPL-3.0-only

#include "cuajone/capture.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

namespace cuajone {

LatestFrameCapture::LatestFrameCapture(
    std::string source,
    std::chrono::duration<double> reconnect_delay,
    std::chrono::duration<double> maximum_reconnect_delay,
    std::chrono::milliseconds open_timeout,
    std::chrono::milliseconds read_timeout,
    RtspTransport rtsp_transport)
    : source_(std::move(source)),
      reconnect_delay_(reconnect_delay),
      maximum_reconnect_delay_(maximum_reconnect_delay),
      open_timeout_(open_timeout),
      read_timeout_(read_timeout),
      rtsp_transport_(rtsp_transport) {
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
    std::chrono::milliseconds timeout) {
    if (timeout.count() < 0) throw std::invalid_argument("Frame wait timeout must be non-negative");
    std::unique_lock lock(mutex_);
    condition_.wait_for(lock, timeout, [&] {
        return sequence_ != previous_sequence || ended_;
    });
    if (sequence_ == previous_sequence) return false;
    frame = latest_;
    sequence = sequence_;
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
        error_.reset();
    }
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
        std::vector<int> parameters;
        if (open_timeout_.count() > 0) {
            parameters.insert(parameters.end(), {
                cv::CAP_PROP_OPEN_TIMEOUT_MSEC, static_cast<int>(open_timeout_.count())});
        }
        if (read_timeout_.count() > 0) {
            parameters.insert(parameters.end(), {
                cv::CAP_PROP_READ_TIMEOUT_MSEC, static_cast<int>(read_timeout_.count())});
        }
        bool opened = parameters.empty()
            ? capture.open(source_, cv::CAP_FFMPEG)
            : capture.open(source_, cv::CAP_FFMPEG, parameters);
        if (!opened && !parameters.empty()) {
            capture.release();
            opened = capture.open(source_, cv::CAP_FFMPEG);
        }
        if (!opened) {
            {
                std::scoped_lock lock(mutex_);
                error_ = "OpenCV could not open the source";
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
