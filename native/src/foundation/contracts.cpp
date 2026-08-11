// SPDX-License-Identifier: AGPL-3.0-only

#include "cuajone/contracts.hpp"
#include "cuajone/ppe_analytics.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <locale>
#include <sstream>
#include <stdexcept>

namespace cuajone {
namespace {

std::string escapeJson(std::string_view value) {
    std::string escaped;
    escaped.reserve(value.size() + 8);
    for (const unsigned char character : value) {
        switch (character) {
        case '"': escaped += "\\\""; break;
        case '\\': escaped += "\\\\"; break;
        case '\b': escaped += "\\b"; break;
        case '\f': escaped += "\\f"; break;
        case '\n': escaped += "\\n"; break;
        case '\r': escaped += "\\r"; break;
        case '\t': escaped += "\\t"; break;
        default:
            if (character < 0x20U) {
                std::ostringstream code;
                code << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                     << static_cast<int>(character);
                escaped += code.str();
            } else {
                escaped.push_back(static_cast<char>(character));
            }
        }
    }
    return escaped;
}

std::string quote(std::string_view value) {
    return "\"" + escapeJson(value) + "\"";
}

std::string number(float value) {
    if (!std::isfinite(value)) throw std::invalid_argument("Canonical JSON rejects non-finite numbers");
    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << std::fixed << std::setprecision(kCanonicalDecimalDigits) << value;
    std::string text = output.str();
    while (text.size() > 1 && text.back() == '0') text.pop_back();
    if (text.back() == '.') text.pop_back();
    if (text == "-0") text = "0";
    return text;
}

std::string boxJson(const Box& box) {
    return "[" + number(box.x1) + "," + number(box.y1) + ","
        + number(box.x2) + "," + number(box.y2) + "]";
}

std::string keypointsJson(const std::vector<Keypoint>& keypoints) {
    std::string output{"["};
    for (std::size_t index = 0; index < keypoints.size(); ++index) {
        if (index != 0) output += ',';
        const auto& point = keypoints[index];
        output += "[" + number(point.x) + "," + number(point.y) + ","
            + number(point.confidence) + "]";
    }
    return output + "]";
}

std::string stringArrayJson(const std::vector<std::string>& values) {
    std::string output{"["};
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) output += ',';
        output += quote(values[index]);
    }
    return output + "]";
}

std::string ppeJson(const std::optional<PpeEvaluation>& value) {
    if (!value) return "null";
    std::vector<std::string> required;
    std::vector<std::string> present;
    std::vector<std::string> missing;
    std::string items{"["};
    for (std::size_t index = 0; index < value->items.size(); ++index) {
        const auto& item = value->items[index];
        const std::string semantic(ppeItemSemantic(item.item));
        if (item.required) required.push_back(semantic);
        if (value->evaluated) (item.present ? present : missing).push_back(semantic);
        if (index != 0) items += ',';
        items += "{\"confidence\":" + number(item.confidence)
            + ",\"detection\":" + (item.detection
                ? "{\"box\":" + boxJson(item.detection->box) + ",\"confidence\":"
                    + number(item.detection->confidence) + "}"
                : "null")
            + ",\"label\":" + quote(ppeItemLabel(item.item))
            + ",\"present\":" + (item.present ? "true" : "false")
            + ",\"ratio\":" + number(item.ratio)
            + ",\"required\":" + (item.required ? "true" : "false")
            + ",\"semantic\":" + quote(semantic) + "}";
    }
    items += ']';
    const std::string state = !value->evaluated ? "evaluating"
        : value->compliant ? "compliant" : "noncompliant";
    return "{\"items\":" + items + ",\"missing\":" + stringArrayJson(missing)
        + ",\"present\":" + stringArrayJson(present)
        + ",\"required\":" + stringArrayJson(required)
        + ",\"samples\":" + std::to_string(value->samples)
        + ",\"state\":" + quote(state) + "}";
}

std::string legacyEventType(std::string_view type) {
    if (type == "com.cuajone.safety.ppe.violation.v2") return "com.cuajone.safety.ppe.violation.v1";
    if (type == "com.cuajone.safety.fall.possible.v2") return "com.cuajone.safety.fall.possible.v1";
    return std::string(type);
}

std::string legacyStatus(const CanonicalEvent& event) {
    if (event.ppe) return event.ppe->evaluated ? legacyPpeStatus(*event.ppe) : "En evaluación";
    return event.status == "Evaluando EPP" ? "En evaluación" : event.status;
}

bool containsSecretMaterial(std::string_view value) {
    std::string lower(value);
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return lower.find("rtsp://") != std::string::npos
        || lower.find("rtsps://") != std::string::npos
        || lower.find("password") != std::string::npos
        || lower.find('@') != std::string::npos;
}

}  // namespace

void validateContractVersion(std::string_view version) {
    // No implicit upgrade/downgrade: every producer must opt into the exact schema.
    if (version != kContractVersion) {
        throw std::invalid_argument(
            "Unsupported contract version '" + std::string(version)
            + "'; expected " + std::string(kContractVersion));
    }
}

void validateCanonicalMetadata(const CanonicalFrameResult& result) {
    if (result.source_id.empty() || containsSecretMaterial(result.source_id)) {
        throw std::invalid_argument("Canonical source_id must be non-secret and must not contain an RTSP URL");
    }
    if (result.observed_at.empty()) {
        throw std::invalid_argument("Canonical observed_at is required");
    }
    if (result.monotonic_timestamp_ms < 0 || result.frame_width <= 0 || result.frame_height <= 0) {
        throw std::invalid_argument("Canonical timestamp and frame dimensions are outside supported ranges");
    }
}

std::string canonicalJson(const CanonicalEvent& event) {
    return "{\"contractversion\":\"1.0.0\",\"data\":{\"confidence\":"
        + number(event.confidence) + ",\"contract_version\":\"1.0.0\",\"evidence\":[],\"frame_id\":"
        + std::to_string(event.frame_id) + ",\"monotonic_timestamp_ms\":"
        + std::to_string(event.monotonic_timestamp_ms) + ",\"status\":" + quote(legacyStatus(event))
        + ",\"track_id\":" + std::to_string(event.track_id)
        + "},\"datacontenttype\":\"application/json\",\"dataschema\":\"https://cuajone.example/contracts/v1/event.schema.json\",\"id\":"
        + quote(event.id) + ",\"source\":" + quote(event.source)
        + ",\"specversion\":\"1.0\",\"subject\":" + quote(event.subject)
        + ",\"time\":" + quote(event.time) + ",\"type\":" + quote(legacyEventType(event.type)) + "}";
}

std::string canonicalJson(const CanonicalFrameResult& result) {
    validateCanonicalMetadata(result);
    std::string output = "{\"contract_version\":\"1.0.0\",\"events\":[";
    for (std::size_t index = 0; index < result.events.size(); ++index) {
        if (index != 0) output += ',';
        output += quote(result.events[index].id);
    }
    output += "],\"frame\":{\"height\":" + std::to_string(result.frame_height)
        + ",\"width\":" + std::to_string(result.frame_width) + "},\"frame_id\":"
        + std::to_string(result.frame_id) + ",\"monotonic_timestamp_ms\":"
        + std::to_string(result.monotonic_timestamp_ms) + ",\"observed_at\":"
        + quote(result.observed_at) + ",\"people\":[";
    for (std::size_t index = 0; index < result.people.size(); ++index) {
        if (index != 0) output += ',';
        const auto& person = result.people[index];
        output += "{\"box\":" + boxJson(person.box) + ",\"confidence\":"
            + number(person.confidence) + ",\"fall_active\":"
            + (person.fall_active ? "true" : "false") + ",\"keypoints\":"
            + keypointsJson(person.keypoints) + ",\"ppe_evaluable\":"
            + (person.ppe_evaluable ? "true" : "false") + ",\"ppe_status\":"
            + quote(person.ppe ? legacyPpeStatus(*person.ppe) : person.ppe_status) + ",\"track_id\":"
            + std::to_string(person.track_id) + "}";
    }
    return output + "],\"source_id\":" + quote(result.source_id) + "}";
}

std::string canonicalJsonV2(const CanonicalEvent& event) {
    return "{\"contractversion\":\"2.0.0\",\"data\":{\"confidence\":"
        + number(event.confidence) + ",\"contract_version\":\"2.0.0\",\"evidence\":[],\"frame_id\":"
        + std::to_string(event.frame_id) + ",\"monotonic_timestamp_ms\":"
        + std::to_string(event.monotonic_timestamp_ms) + ",\"ppe\":" + ppeJson(event.ppe)
        + ",\"status\":" + quote(event.status) + ",\"track_id\":" + std::to_string(event.track_id)
        + "},\"datacontenttype\":\"application/json\",\"dataschema\":\"https://cuajone.example/contracts/v2/event.schema.json\",\"id\":"
        + quote(event.id) + ",\"source\":" + quote(event.source)
        + ",\"specversion\":\"1.0\",\"subject\":" + quote(event.subject)
        + ",\"time\":" + quote(event.time) + ",\"type\":" + quote(event.type) + "}";
}

std::string canonicalJsonV2(const CanonicalFrameResult& result) {
    validateCanonicalMetadata(result);
    std::string output = "{\"contract_version\":\"2.0.0\",\"events\":[";
    for (std::size_t index = 0; index < result.events.size(); ++index) {
        if (index != 0) output += ',';
        output += quote(result.events[index].id);
    }
    output += "],\"frame\":{\"height\":" + std::to_string(result.frame_height)
        + ",\"width\":" + std::to_string(result.frame_width) + "},\"frame_id\":"
        + std::to_string(result.frame_id) + ",\"monotonic_timestamp_ms\":"
        + std::to_string(result.monotonic_timestamp_ms) + ",\"observed_at\":"
        + quote(result.observed_at) + ",\"people\":[";
    for (std::size_t index = 0; index < result.people.size(); ++index) {
        if (index != 0) output += ',';
        const auto& person = result.people[index];
        output += "{\"box\":" + boxJson(person.box) + ",\"confidence\":"
            + number(person.confidence) + ",\"fall_active\":"
            + (person.fall_active ? "true" : "false") + ",\"keypoints\":"
            + keypointsJson(person.keypoints) + ",\"ppe\":" + ppeJson(person.ppe)
            + ",\"ppe_visibility_sufficient\":" + (person.ppe_evaluable ? "true" : "false")
            + ",\"track_id\":" + std::to_string(person.track_id) + "}";
    }
    return output + "],\"source_id\":" + quote(result.source_id) + "}";
}

std::string runtimeDefaultsJson() {
    return R"({"analytics":{"backend":"native","mode":"ppe-fall"},"contract_version":"1.0.0","fall":{"alert_cooldown_ms":120000,"aspect_ratio":1.05,"confirm_frames":12,"descent_ratio":0.12,"near_floor_ratio":0.65,"reset_frames":20,"torso_angle_degrees":55.0,"track_ttl_ms":5000},"ppe":{"alert_cooldown_ms":60000,"minimum_samples":12,"present_ratio":0.35,"track_ttl_ms":5000,"window":20},"thresholds":{"maximum_detections":300,"nms_iou":0.45,"pose_confidence":0.35,"ppe_confidence":0.3},"tracker":{"frame_rate":30,"high_confidence_threshold":0.35,"low_confidence_threshold":0.1,"match_threshold":0.8,"maximum_age":30,"maximum_tracks":128,"profile":"byte-track-eigen"}})";
}

}  // namespace cuajone
