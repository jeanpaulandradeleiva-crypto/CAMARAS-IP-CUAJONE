// SPDX-License-Identifier: AGPL-3.0-only

#include "cuajone/analytics_pipeline.hpp"
#include "cuajone/letterbox.hpp"
#ifdef CUAJONE_PYTHON_WITH_ENGINE_RUNTIME
#include "cuajone/engine_pipeline.hpp"
#include <opencv2/core/mat.hpp>
#endif

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

namespace py = pybind11;
using namespace cuajone;

namespace {

std::string processObservations(AnalyticsPipeline& pipeline, const ObservationFrame& frame) {
    ProcessedFrame result;
    {
        // Serialize after reacquiring the GIL; this block accesses native state only.
        py::gil_scoped_release release;
        result = pipeline.process(frame);
    }
    return canonicalJson(result.canonical);
}

std::pair<std::string, std::vector<std::string>> processObservationBundle(
    AnalyticsPipeline& pipeline,
    const ObservationFrame& frame) {
    ProcessedFrame result;
    {
        py::gil_scoped_release release;
        result = pipeline.process(frame);
    }
    std::vector<std::string> events;
    events.reserve(result.canonical.events.size());
    for (const auto& event : result.canonical.events) events.push_back(canonicalJson(event));
    return {canonicalJson(result.canonical), std::move(events)};
}

void applyFrameDimensions(const py::array& bgr_frame, ObservationFrame& observations) {
    // Reject layouts that would require a hidden copy before exposing borrowed memory.
    if (!bgr_frame.dtype().is(py::dtype::of<std::uint8_t>())) {
        throw py::type_error("frame must have dtype numpy.uint8");
    }
    if (bgr_frame.ndim() != 3 || bgr_frame.shape(2) != 3) {
        throw py::value_error("frame must have BGR shape (height, width, 3)");
    }
    if ((bgr_frame.flags() & py::array::c_style) == 0) {
        throw py::value_error("frame must be C-contiguous; implicit copies are not allowed");
    }
    if (bgr_frame.shape(0) <= 0 || bgr_frame.shape(1) <= 0) {
        throw py::value_error("frame dimensions must be positive");
    }
    observations.frame_height = static_cast<int>(bgr_frame.shape(0));
    observations.frame_width = static_cast<int>(bgr_frame.shape(1));
}

std::string processFrame(
    AnalyticsPipeline& pipeline,
    const py::array& bgr_frame,
    ObservationFrame observations) {
    applyFrameDimensions(bgr_frame, observations);
    return processObservations(pipeline, observations);
}

std::pair<std::string, std::vector<std::string>> processFrameBundle(
    AnalyticsPipeline& pipeline,
    const py::array& bgr_frame,
    ObservationFrame observations) {
    applyFrameDimensions(bgr_frame, observations);
    return processObservationBundle(pipeline, observations);
}

#ifdef CUAJONE_PYTHON_WITH_ENGINE_RUNTIME
std::pair<std::string, std::vector<std::string>> processEngineFrame(
    NativeEnginePipeline& pipeline,
    const py::array& bgr_frame,
    const std::string& source_id,
    std::uint64_t frame_id,
    std::int64_t monotonic_timestamp_ms,
    const std::string& observed_at) {
    ObservationFrame dimensions;
    applyFrameDimensions(bgr_frame, dimensions);
    // cv::Mat borrows the NumPy buffer; bgr_frame owns it for this synchronous call.
    cv::Mat view(
        dimensions.frame_height,
        dimensions.frame_width,
        CV_8UC3,
        const_cast<void*>(bgr_frame.data()),
        static_cast<std::size_t>(bgr_frame.strides(0)));
    ProcessedFrame result;
    {
        // C++ consumes the borrowed view synchronously and touches no Python objects here.
        py::gil_scoped_release release;
        result = pipeline.processFrame(
            view, source_id, frame_id, monotonic_timestamp_ms, observed_at);
    }
    std::vector<std::string> events;
    events.reserve(result.canonical.events.size());
    for (const auto& event : result.canonical.events) events.push_back(canonicalJson(event));
    return {canonicalJson(result.canonical), std::move(events)};
}
#endif

}  // namespace

PYBIND11_MODULE(cuajone_native, module) {
    module.doc() = "Development/QA bindings for the deterministic Cuajone C++ core";
    module.attr("__version__") = std::string(kRuntimeVersion);
    module.attr("CONTRACT_VERSION") = std::string(kContractVersion);

    py::enum_<AnalyticsMode>(module, "AnalyticsMode")
        .value("PPE_ONLY", AnalyticsMode::PpeOnly)
        .value("PPE_FALL", AnalyticsMode::PpeFall);

    py::class_<Box>(module, "Box")
        .def(py::init<>())
        .def_readwrite("x1", &Box::x1)
        .def_readwrite("y1", &Box::y1)
        .def_readwrite("x2", &Box::x2)
        .def_readwrite("y2", &Box::y2);
    py::class_<Keypoint>(module, "Keypoint")
        .def(py::init<>())
        .def_readwrite("x", &Keypoint::x)
        .def_readwrite("y", &Keypoint::y)
        .def_readwrite("confidence", &Keypoint::confidence);
    py::class_<Detection>(module, "Detection")
        .def(py::init<>())
        .def_readwrite("box", &Detection::box)
        .def_readwrite("confidence", &Detection::confidence)
        .def_readwrite("class_id", &Detection::class_id);
    py::class_<PoseDetection, Detection>(module, "PoseDetection")
        .def(py::init<>())
        .def_readwrite("keypoints", &PoseDetection::keypoints);
    py::class_<PpeClassMap>(module, "PpeClassMap")
        .def(py::init<>())
        .def_readwrite("person_ids", &PpeClassMap::person_ids)
        .def_readwrite("helmet_ids", &PpeClassMap::helmet_ids)
        .def_readwrite("vest_ids", &PpeClassMap::vest_ids);
    py::class_<IoUTrackerConfig>(module, "TrackerConfig")
        .def(py::init<>())
        .def_readwrite("minimum_iou", &IoUTrackerConfig::minimum_iou)
        .def_readwrite("maximum_age", &IoUTrackerConfig::maximum_age)
        .def_readwrite("maximum_tracks", &IoUTrackerConfig::maximum_tracks);
    py::class_<PpeConfig>(module, "PpeConfig")
        .def(py::init<>())
        .def_property("window", [](const PpeConfig& value) { return value.window; },
            [](PpeConfig& value, std::size_t input) { value.window = input; })
        .def_property("minimum_samples", [](const PpeConfig& value) { return value.minimum_samples; },
            [](PpeConfig& value, std::size_t input) { value.minimum_samples = input; })
        .def_readwrite("present_ratio", &PpeConfig::present_ratio)
        .def_property("alert_cooldown_ms",
            [](const PpeConfig& value) { return value.alert_cooldown.count() * 1000.0; },
            [](PpeConfig& value, double input) { value.alert_cooldown = std::chrono::duration<double>(input / 1000.0); })
        .def_property("track_ttl_ms",
            [](const PpeConfig& value) { return value.track_ttl.count() * 1000.0; },
            [](PpeConfig& value, double input) { value.track_ttl = std::chrono::duration<double>(input / 1000.0); });
    py::class_<FallConfig>(module, "FallConfig")
        .def(py::init<>())
        .def_readwrite("confirm_frames", &FallConfig::confirm_frames)
        .def_readwrite("reset_frames", &FallConfig::reset_frames)
        .def_property("alert_cooldown_ms",
            [](const FallConfig& value) { return value.alert_cooldown.count() * 1000.0; },
            [](FallConfig& value, double input) { value.alert_cooldown = std::chrono::duration<double>(input / 1000.0); })
        .def_property("track_ttl_ms",
            [](const FallConfig& value) { return value.track_ttl.count() * 1000.0; },
            [](FallConfig& value, double input) { value.track_ttl = std::chrono::duration<double>(input / 1000.0); })
        .def_readwrite("aspect_ratio", &FallConfig::aspect_ratio)
        .def_readwrite("torso_angle_degrees", &FallConfig::torso_angle_degrees)
        .def_readwrite("descent_ratio", &FallConfig::descent_ratio)
        .def_readwrite("near_floor_ratio", &FallConfig::near_floor_ratio);
    py::class_<AnalyticsPipelineConfig>(module, "AnalyticsConfig")
        .def(py::init<>())
        .def_readwrite("mode", &AnalyticsPipelineConfig::mode)
        .def_readwrite("tracker", &AnalyticsPipelineConfig::tracker)
        .def_readwrite("ppe", &AnalyticsPipelineConfig::ppe)
        .def_readwrite("fall", &AnalyticsPipelineConfig::fall)
        .def_readwrite("pose_confidence", &AnalyticsPipelineConfig::pose_confidence)
        .def_readwrite("nms_iou", &AnalyticsPipelineConfig::nms_iou);
    py::class_<ObservationFrame>(module, "ObservationFrame")
        .def(py::init<>())
        .def_readwrite("contract_version", &ObservationFrame::contract_version)
        .def_readwrite("source_id", &ObservationFrame::source_id)
        .def_readwrite("frame_id", &ObservationFrame::frame_id)
        .def_readwrite("monotonic_timestamp_ms", &ObservationFrame::monotonic_timestamp_ms)
        .def_readwrite("observed_at", &ObservationFrame::observed_at)
        .def_readwrite("frame_width", &ObservationFrame::frame_width)
        .def_readwrite("frame_height", &ObservationFrame::frame_height)
        .def_readwrite("ppe_detections", &ObservationFrame::ppe_detections)
        .def_readwrite("pose_detections", &ObservationFrame::pose_detections)
        .def_readwrite("ppe_classes", &ObservationFrame::ppe_classes);
    // Python owns each pipeline instance; wrappers borrow it only for a synchronous call.
    py::class_<AnalyticsPipeline>(module, "AnalyticsPipeline")
        .def(py::init<AnalyticsPipelineConfig>(), py::arg("config") = AnalyticsPipelineConfig{})
        .def("process_observations", &processObservations, py::arg("observations"))
        .def("process_observations_bundle", &processObservationBundle, py::arg("observations"))
        .def("process_frame", &processFrame, py::arg("frame"), py::arg("observations"))
        .def("process_frame_bundle", &processFrameBundle, py::arg("frame"), py::arg("observations"))
        .def("reset", &AnalyticsPipeline::reset)
        .def_property_readonly("contract_version", [](const AnalyticsPipeline& value) {
            return std::string(value.contractVersion());
        })
        .def_property_readonly("runtime_version", [](const AnalyticsPipeline& value) {
            return std::string(value.runtimeVersion());
        });

#ifdef CUAJONE_PYTHON_WITH_ENGINE_RUNTIME
    py::class_<EnginePipelineConfig>(module, "EngineConfig")
        .def(py::init<>())
        .def_property("ppe_engine",
            [](const EnginePipelineConfig& value) { return value.ppe_engine.string(); },
            [](EnginePipelineConfig& value, const std::string& input) { value.ppe_engine = input; })
        .def_property("pose_engine",
            [](const EnginePipelineConfig& value) { return value.pose_engine.string(); },
            [](EnginePipelineConfig& value, const std::string& input) { value.pose_engine = input; })
        .def_readwrite("ppe_labels", &EnginePipelineConfig::ppe_labels)
        .def_readwrite("pose_class_count", &EnginePipelineConfig::pose_class_count)
        .def_readwrite("pose_keypoint_shape", &EnginePipelineConfig::pose_keypoint_shape)
        .def_readwrite("allow_nonperson_pose_class", &EnginePipelineConfig::allow_nonperson_pose_class)
        .def_readwrite("device", &EnginePipelineConfig::device)
        .def_readwrite("ppe_confidence", &EnginePipelineConfig::ppe_confidence)
        .def_readwrite("pose_confidence", &EnginePipelineConfig::pose_confidence)
        .def_readwrite("nms_iou", &EnginePipelineConfig::nms_iou)
        .def_readwrite("maximum_detections", &EnginePipelineConfig::maximum_detections)
        .def_readwrite("analytics", &EnginePipelineConfig::analytics);
    py::class_<NativeEnginePipeline>(module, "EnginePipeline")
        .def(py::init<EnginePipelineConfig>())
        .def("process_frame", &processEngineFrame,
            py::arg("frame"), py::arg("source_id"), py::arg("frame_id"),
            py::arg("monotonic_timestamp_ms"), py::arg("observed_at"))
        .def("reset", &NativeEnginePipeline::reset);
    module.attr("ENGINE_RUNTIME_AVAILABLE") = true;
#else
    module.attr("ENGINE_RUNTIME_AVAILABLE") = false;
#endif

    module.def("runtime_defaults_json", &runtimeDefaultsJson);
    module.def("letterbox_transform", [](int source_width, int source_height, int model_width, int model_height) {
        const auto value = makeLetterboxTransform(source_width, source_height, model_width, model_height);
        py::dict result;
        result["scale_x"] = value.scale_x;
        result["scale_y"] = value.scale_y;
        result["padding_left"] = value.padding_left;
        result["padding_top"] = value.padding_top;
        return result;
    });
}
