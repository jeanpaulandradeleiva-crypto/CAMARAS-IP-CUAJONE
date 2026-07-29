// SPDX-License-Identifier: AGPL-3.0-only

#include "cuajone/capture.hpp"
#include "cuajone/cli.hpp"
#include "cuajone/engine_reader.hpp"
#include "cuajone/evidence.hpp"
#include "cuajone/fall_analytics.hpp"
#include "cuajone/iou_tracker.hpp"
#include "cuajone/ppe_analytics.hpp"
#include "cuajone/preprocess.hpp"
#include "cuajone/tensorrt_runtime.hpp"
#include "cuajone/yolo_decode.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <filesystem>
#include <iostream>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace cuajone;
using Clock = std::chrono::steady_clock;

std::atomic_bool stop_requested{};

void requestStop(int) {
    stop_requested.store(true, std::memory_order_relaxed);
}

struct PreparedEngines {
    EngineFile ppe;
    EngineFile pose;
    std::map<int, std::string> ppe_names;
    PpeClassMap ppe_classes;
    std::size_t pose_class_count{};
    std::array<int, 2> keypoint_shape{};
};

void validateContiguousNames(const std::map<int, std::string>& names, const std::string& engine_name) {
    if (names.empty()) throw std::runtime_error(engine_name + " requires class names metadata or an explicit fallback");
    int expected = 0;
    for (const auto& [id, name] : names) {
        if (id != expected++ || name.empty()) {
            throw std::runtime_error(engine_name + " class IDs must be contiguous from zero");
        }
    }
}

void validateTask(const EngineMetadata& metadata, const std::string& expected, const std::string& engine_name) {
    if (metadata.task && normalizeLabel(*metadata.task) != expected) {
        throw std::runtime_error(engine_name + " metadata task is '" + *metadata.task
            + "', expected '" + expected + "'");
    }
}

PreparedEngines prepareEngineFiles(const RuntimeConfig& config) {
    if (!std::filesystem::is_regular_file(config.ppe_engine)) {
        throw std::runtime_error("PPE engine does not exist: " + config.ppe_engine.string());
    }
    if (!std::filesystem::is_regular_file(config.pose_engine)) {
        throw std::runtime_error("Pose engine does not exist: " + config.pose_engine.string());
    }
    EngineFile ppe = EngineFile::read(config.ppe_engine);
    EngineFile pose = EngineFile::read(config.pose_engine);
    validateTask(ppe.metadata(), "detect", "PPE engine");
    validateTask(pose.metadata(), "pose", "Pose engine");

    std::map<int, std::string> ppe_names = ppe.metadata().names;
    if (ppe_names.empty() && config.ppe_labels) ppe_names = *config.ppe_labels;
    validateContiguousNames(ppe_names, "PPE engine");
    PpeClassMap ppe_classes = resolvePpeClasses(ppe_names);

    std::size_t pose_class_count = pose.metadata().names.empty()
        ? config.pose_class_count : pose.metadata().names.size();
    if (!pose.metadata().names.empty()) validateContiguousNames(pose.metadata().names, "Pose engine");
    if (pose_class_count != 1) {
        throw std::runtime_error("The first native pose candidate supports exactly one person class");
    }
    if (!pose.metadata().names.empty()) {
        if (!isPersonClassLabel(pose.metadata().names.begin()->second)
            && !config.allow_nonperson_pose_class) {
            throw std::runtime_error(
                "Pose engine's single metadata class must normalize to person/persona; "
                "use --allow-nonperson-pose-class only after verifying the engine contract");
        }
    }
    std::array<int, 2> keypoint_shape = config.pose_keypoint_shape;
    if (pose.metadata().keypoint_shape) keypoint_shape = *pose.metadata().keypoint_shape;
    if (keypoint_shape[1] < 3) throw std::runtime_error("Pose keypoint dimensions must include x, y, and confidence");
    return {
        std::move(ppe), std::move(pose), std::move(ppe_names), std::move(ppe_classes),
        pose_class_count, keypoint_shape,
    };
}

void validateSourceWithoutOpening(const std::string& source) {
    const bool network = isRtspSource(source);
    if (network) {
        validateRtspSource(source);
        return;
    }
    if (!std::filesystem::is_regular_file(source)) {
        throw std::runtime_error("Offline source does not exist: " + source);
    }
}

PreparedEngines runBasePreflight(const RuntimeConfig& config) {
    validateSourceWithoutOpening(config.source);
    validateWritableOutput(config.output);
    const DeviceSummary device = selectCudaDevice(config.device);
    std::cout << "OpenCV: " << CV_VERSION << " | TensorRT headers: "
              << NV_TENSORRT_MAJOR << '.' << NV_TENSORRT_MINOR << '\n';
    std::cout << "CUDA device " << device.selected << ": " << device.name
              << " | SM " << device.compute_major << '.' << device.compute_minor
              << " | devices: " << device.count << '\n';

    PreparedEngines prepared = prepareEngineFiles(config);
    std::cout << "PPE engine: " << config.ppe_engine.string()
              << " | metadata prefix: " << (prepared.ppe.hasMetadataPrefix() ? "yes" : "no") << '\n';
    std::cout << "Pose engine: " << config.pose_engine.string()
              << " | metadata prefix: " << (prepared.pose.hasMetadataPrefix() ? "yes" : "no") << '\n';
    std::cout << "Source: " << redactSource(config.source) << " | label: " << config.source_label << '\n';
    std::cout << "Output: " << config.output.string() << '\n';
    return prepared;
}

void drawPose(cv::Mat& frame, const PoseDetection& pose, float threshold) {
    static constexpr std::array<std::array<int, 2>, 16> skeleton{{
        {0, 1}, {0, 2}, {1, 3}, {2, 4}, {5, 6}, {5, 7}, {7, 9}, {6, 8},
        {8, 10}, {5, 11}, {6, 12}, {11, 12}, {11, 13}, {13, 15}, {12, 14}, {14, 16},
    }};
    for (const auto& edge : skeleton) {
        if (edge[0] >= static_cast<int>(pose.keypoints.size()) || edge[1] >= static_cast<int>(pose.keypoints.size())) continue;
        const auto& from = pose.keypoints[edge[0]];
        const auto& to = pose.keypoints[edge[1]];
        if (from.confidence < threshold || to.confidence < threshold) continue;
        cv::line(frame, cv::Point(static_cast<int>(from.x), static_cast<int>(from.y)),
                 cv::Point(static_cast<int>(to.x), static_cast<int>(to.y)),
                 cv::Scalar(255, 140, 0), 2, cv::LINE_AA);
    }
}

void drawPerson(cv::Mat& frame, const TrackedPerson& person, const std::string& status, bool falling) {
    cv::rectangle(
        frame,
        cv::Point(static_cast<int>(person.box.x1), static_cast<int>(person.box.y1)),
        cv::Point(static_cast<int>(person.box.x2), static_cast<int>(person.box.y2)),
        cv::Scalar(255, 255, 255), 2);
    std::string text = "T" + std::to_string(person.track_id) + " | " + status;
    if (falling) text += " | POSSIBLE FALL";
    cv::putText(frame, text,
        cv::Point(static_cast<int>(person.box.x1), std::max(25, static_cast<int>(person.box.y1) - 10)),
        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
}

void drawAssociatedItem(cv::Mat& frame, const std::optional<Detection>& item, const std::string& label) {
    if (!item) return;
    cv::rectangle(frame,
        cv::Point(static_cast<int>(item->box.x1), static_cast<int>(item->box.y1)),
        cv::Point(static_cast<int>(item->box.x2), static_cast<int>(item->box.y2)),
        cv::Scalar(0, 220, 255), 2);
    cv::putText(frame, label, cv::Point(static_cast<int>(item->box.x1), std::max(20, static_cast<int>(item->box.y1) - 6)),
        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 220, 255), 2, cv::LINE_AA);
}

int monitor(
    const RuntimeConfig& config,
    const PreparedEngines& prepared,
    TensorRtSession& ppe_session,
    TensorRtSession& pose_session) {
    LetterboxPreprocessor ppe_preprocessor(
        ppe_session.inputWidth(), ppe_session.inputHeight());
    LetterboxPreprocessor pose_preprocessor(
        pose_session.inputWidth(), pose_session.inputHeight());
    IoUTracker tracker({config.tracker_iou, config.tracker_max_age, config.tracker_max_tracks});
    PpeAnalyzer ppe_analyzer(config.ppe);
    FallAnalyzer fall_analyzer(config.fall);
    EvidenceWriter evidence(config.output);
    LatestFrameCapture capture(
        config.source,
        std::chrono::duration<double>(config.reconnect_delay_seconds),
        std::chrono::duration<double>(config.maximum_reconnect_delay_seconds),
        config.capture_open_timeout,
        config.capture_read_timeout,
        config.rtsp_transport);

    capture.start();
    std::uint64_t sequence{};
    auto next_inference = Clock::time_point::min();
    std::optional<std::string> last_capture_error;
    if (config.show_window) cv::namedWindow("Cuajone native analytics", cv::WINDOW_NORMAL);

    while (!stop_requested.load(std::memory_order_relaxed)) {
        cv::Mat frame;
        std::uint64_t latest_sequence = sequence;
        if (!capture.waitForLatest(sequence, frame, latest_sequence, std::chrono::milliseconds(100))) {
            if (capture.ended()) break;
            const auto error = capture.lastError();
            if (error && error != last_capture_error) {
                std::cerr << "Capture: " << *error << '\n';
                last_capture_error = error;
            }
            continue;
        }
        sequence = latest_sequence;
        const auto now = Clock::now();
        if (config.target_fps > 0.0 && now < next_inference) continue;
        if (config.target_fps > 0.0) {
            next_inference = now + std::chrono::duration_cast<Clock::duration>(
                std::chrono::duration<double>(1.0 / config.target_fps));
        }

        // The two engines share no work queues: PPE completes before pose starts.
        const auto ppe_input = ppe_preprocessor.process(frame);
        const auto ppe_output = ppe_session.infer(ppe_input.nchw);
        auto ppe_detections = decodeDetections(
            {ppe_output.values, ppe_output.shape},
            prepared.ppe_names.size(), config.ppe_confidence, config.nms_iou,
            ppe_input.transform, {DecodeLimits{}.max_nms_candidates, config.max_det});

        const auto pose_input = pose_preprocessor.process(frame);
        const auto pose_output = pose_session.infer(pose_input.nchw);
        auto poses = decodePoses(
            {pose_output.values, pose_output.shape}, prepared.pose_class_count,
            static_cast<std::size_t>(prepared.keypoint_shape[0]),
            static_cast<std::size_t>(prepared.keypoint_shape[1]),
            config.pose_confidence, config.nms_iou, pose_input.transform,
            {DecodeLimits{}.max_nms_candidates, config.max_det});

        std::vector<Box> pose_boxes;
        pose_boxes.reserve(poses.size());
        for (const auto& pose : poses) pose_boxes.push_back(pose.box);
        const auto track_ids = tracker.update(pose_boxes);

        std::vector<Box> ppe_person_boxes;
        for (const auto& detection : ppe_detections) {
            if (std::find(prepared.ppe_classes.person_ids.begin(),
                          prepared.ppe_classes.person_ids.end(), detection.class_id)
                != prepared.ppe_classes.person_ids.end()) {
                ppe_person_boxes.push_back(detection.box);
            }
        }

        const float keypoint_threshold = std::clamp(config.pose_confidence, 0.25F, 0.50F);
        std::vector<TrackedPerson> people;
        std::vector<std::size_t> pose_indices;
        for (std::size_t index = 0; index < poses.size(); ++index) {
            if (track_ids[index] < 0 || !isValidPosePerson(
                    poses[index], ppe_person_boxes, config.pose_confidence, config.nms_iou)) {
                continue;
            }
            const bool evaluable = isBoxPpeEvaluable(poses[index].box, frame.cols, frame.rows)
                && arePoseKeypointsPpeEvaluable(poses[index].keypoints, keypoint_threshold);
            people.push_back({
                track_ids[index], poses[index].box, poses[index].confidence,
                poses[index].keypoints, evaluable,
            });
            pose_indices.push_back(index);
        }
        const auto associations = associatePpe(people, ppe_detections, prepared.ppe_classes);
        std::vector<EventCandidate> pending_events;
        for (std::size_t index = 0; index < people.size(); ++index) {
            const auto& person = people[index];
            const auto& association = associations.at(person.track_id);
            std::string status = person.ppe_evaluable ? "Evaluating PPE" : "PPE not evaluable";
            if (auto event = ppe_analyzer.update(
                    person.track_id, association, person.ppe_evaluable, now)) {
                status = event->status;
                pending_events.push_back(std::move(*event));
            }
            if (person.ppe_evaluable) {
                if (const auto stable_status = ppe_analyzer.currentStatus(person.track_id)) {
                    status = *stable_status;
                }
            }
            const FallResult fall = fall_analyzer.update(
                person.track_id, person.box, person.keypoints, frame.rows, now);
            if (fall.confirmed_now) {
                pending_events.push_back({
                    person.track_id, "POSIBLE_CAIDA",
                    person.ppe_evaluable ? status : "En evaluación", fall.score,
                });
            }
            drawPose(frame, poses[pose_indices[index]], keypoint_threshold);
            drawPerson(frame, person, status, fall.active);
            drawAssociatedItem(frame, association.helmet_detection, "helmet");
            drawAssociatedItem(frame, association.vest_detection, "vest");
        }
        ppe_analyzer.prune(now);
        fall_analyzer.prune(now);

        for (const auto& event : pending_events) {
            try {
                const auto record = evidence.append(frame, config.source_label, event);
                std::cout << "Event: " << record.event_type << " | track " << record.track_id
                          << " | " << record.timestamp << '\n';
            } catch (const std::exception& error) {
                std::cerr << "Evidence write failed: " << error.what() << '\n';
            }
        }

        if (config.show_window) {
            cv::imshow("Cuajone native analytics", frame);
            const int key = cv::waitKey(1) & 0xFF;
            if (key == 'q' || key == 27) stop_requested.store(true, std::memory_order_relaxed);
        }
    }
    capture.stop();
    cv::destroyAllWindows();
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const RuntimeConfig config = parseCommandLine(argc, argv);
        if (config.help) {
            printHelp(std::cout);
            return 0;
        }
        std::signal(SIGINT, requestStop);
        std::signal(SIGTERM, requestStop);
        PreparedEngines prepared = runBasePreflight(config);
        TensorRtSession ppe_session(prepared.ppe, prepared.ppe.metadata().image_size);
        validateDetectSchema(ppe_session.outputShape(), prepared.ppe_names.size());
        TensorRtSession pose_session(prepared.pose, prepared.pose.metadata().image_size);
        validatePoseSchema(
            pose_session.outputShape(), prepared.pose_class_count,
            static_cast<std::size_t>(prepared.keypoint_shape[0]),
            static_cast<std::size_t>(prepared.keypoint_shape[1]));
        std::cout << "Preflight: OK\n";
        if (config.preflight) return 0;
        return monitor(config, prepared, ppe_session, pose_session);
    } catch (const std::invalid_argument& error) {
        std::cerr << "Configuration error: " << error.what() << "\nUse --help for usage.\n";
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "Runtime error: " << error.what() << '\n';
        return 1;
    }
}
