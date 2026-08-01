// SPDX-License-Identifier: AGPL-3.0-only

#include "cuajone/runtime_execution_plan.hpp"

namespace cuajone {

ComputeBackend resolveRequestedComputeBackend(
    ComputeBackend configured_backend,
    bool compute_explicit,
    std::optional<ComputeBackend> installed_backend) {
    if (!compute_explicit && installed_backend) return *installed_backend;
    return configured_backend;
}

RuntimeModelRequirements resolveRuntimeModelRequirements(
    AnalyticsMode analytics_mode,
    const ModelArtifactAvailability& models) noexcept {
    const bool pose_required = analytics_mode == AnalyticsMode::PpeFall;
    return {
        pose_required,
        models.ppe_engine_available && (!pose_required || models.pose_engine_available),
        models.ppe_onnx_available && (!pose_required || models.pose_onnx_available),
    };
}

RuntimeExecutionPlan planRuntimeExecution(const RuntimeExecutionPlanningInput& input) {
    const ComputeBackend requested_backend = resolveRequestedComputeBackend(
        input.configured_backend, input.compute_explicit, input.installed_backend);
    const RuntimeModelRequirements model_requirements = resolveRuntimeModelRequirements(
        input.analytics_mode, input.models);
    const ComputeSelection selection = selectComputeBackend(requested_backend, {
        input.environment.hardware_status,
        input.environment.tensor_rt_runtime_compiled,
        input.environment.onnx_cuda_execution_provider_compiled,
        model_requirements.tensor_rt_models_available,
        model_requirements.onnx_models_available,
    });

    std::optional<ComputeSelection> fallback;
    if (requested_backend == ComputeBackend::Auto
        && selection.backend == ComputeBackend::Cuda
        && model_requirements.onnx_models_available) {
        fallback = ComputeSelection{
            ComputeBackend::Cpu,
            InferenceProvider::OnnxRuntimeCpu,
            "Auto fallback after CUDA preflight validation failed",
        };
    }
    return {requested_backend, model_requirements, selection, fallback};
}

}  // namespace cuajone
