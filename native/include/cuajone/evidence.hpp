// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include "cuajone/contracts.hpp"

#include <opencv2/core/mat.hpp>

#include <filesystem>
#include <string>
#include <string_view>

namespace cuajone {

inline constexpr std::string_view kOperatorEvidenceContractVersion{"1.0.0"};
inline constexpr std::string_view kOperatorCsvHeader{
    "Evento_ID,Camara,Fecha,Hora,Tipo_Evento,Casco,Chaleco,Estado_EPP,"
    "Confianza_Evento,ID_Seguimiento_Temporal,Estado_Revision,"
    "Identificacion_Humana,Observaciones_Revision,Foto"};

struct EvidenceRecord {
    std::string event_id;
    std::string camera;
    std::string date;
    std::string time;
    std::string event_type;
    std::string helmet;
    std::string vest;
    std::string ppe_status;
    float confidence{};
    int track_id{};
    std::string review_status;
    std::string human_identification;
    std::string review_notes;
    std::filesystem::path image_path;
};

void validateWritableOutput(const std::filesystem::path& output);

class EvidenceWriter {
public:
    explicit EvidenceWriter(std::filesystem::path output);
    EvidenceRecord append(
        const cv::Mat& annotated_frame,
        const std::string& source_label,
        const CanonicalEvent& event);

private:
    std::filesystem::path output_;
    std::filesystem::path images_;
    std::filesystem::path csv_;
    std::uint64_t sequence_{};
};

}  // namespace cuajone
