// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include "cuajone/contracts.hpp"

#include <opencv2/core/mat.hpp>

#include <filesystem>
#include <array>
#include <string>
#include <string_view>

namespace cuajone {

inline constexpr std::string_view kOperatorEvidenceContractVersion{"2.0.0"};
inline constexpr std::string_view kOperatorCsvHeader{
    "Version_Contrato,Evento_ID,Camara,Fecha,Hora,Tipo_Evento,Guantes,Botas_Seguridad,"
    "Chaleco,Respirador,Tapaorejas,Casco,Lentes_Protectores,Faltantes_EPP,Estado_EPP,"
    "Ratio_Guantes,Ratio_Botas_Seguridad,Ratio_Chaleco,Ratio_Respirador,Ratio_Tapaorejas,"
    "Ratio_Casco,Ratio_Lentes_Protectores,Confianza_Guantes,Confianza_Botas_Seguridad,"
    "Confianza_Chaleco,Confianza_Respirador,Confianza_Tapaorejas,Confianza_Casco,"
    "Confianza_Lentes_Protectores,"
    "Confianza_Evento,ID_Seguimiento_Temporal,Estado_Revision,"
    "Identificacion_Humana,Observaciones_Revision,Foto"};

struct EvidenceRecord {
    std::string event_id;
    std::string camera;
    std::string date;
    std::string time;
    std::string event_type;
    std::array<std::string, kPpeItemCount> ppe_states;
    std::array<float, kPpeItemCount> ppe_ratios{};
    std::array<float, kPpeItemCount> ppe_confidences{};
    std::string missing_items;
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
