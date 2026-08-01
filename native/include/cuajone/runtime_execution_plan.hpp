// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include "cuajone/compute.hpp"
#include "cuajone/types.hpp"

#include <optional>

namespace cuajone {

struct ModelArtifactAvailability {
    bool ppe_engine_available{};
    bool pose_engine_available{};
    bool ppe_onnx_available{};
    bool pose_onnx_available{};
};

struct RuntimeExecutionEnvironment {
    HardwareProbeStatus hardware_status{HardwareProbeStatus::ProbeError};
    bool tensor_rt_runtime_compiled{};
    bool onnx_cuda_execution_provider_compiled{};
};

struct RuntimeExecutionPlanningInput {
    ComputeBackend configured_backend{ComputeBackend::Auto};
    bool compute_explicit{};
    std::optional<ComputeBackend> installed_backend;
    AnalyticsMode analytics_mode{AnalyticsMode::PpeFall};
    RuntimeExecutionEnvironment environment;
    ModelArtifactAvailability models;
};

struct RuntimeModelRequirements {
    bool pose_required{};
    bool tensor_rt_models_available{};
    bool onnx_models_available{};
};

struct RuntimeExecutionPlan {
    ComputeBackend requested_backend{ComputeBackend::Auto};
    RuntimeModelRequirements model_requirements;
    ComputeSelection selection;
    std::optional<ComputeSelection> preflight_failure_fallback;
};

ComputeBackend resolveRequestedComputeBackend(
    ComputeBackend configured_backend,
    bool compute_explicit,
    std::optional<ComputeBackend> installed_backend);
RuntimeModelRequirements resolveRuntimeModelRequirements(
    AnalyticsMode analytics_mode,
    const ModelArtifactAvailability& models) noexcept;
RuntimeExecutionPlan planRuntimeExecution(const RuntimeExecutionPlanningInput& input);

}  // namespace cuajone
