# SPDX-License-Identifier: AGPL-3.0-only

cmake_minimum_required(VERSION 3.25)

project(cuajone_native LANGUAGES CXX)

option(CUAJONE_BUILD_RUNTIME "Build the Windows native runtime" ON)
option(CUAJONE_BUILD_TESTS "Build pure CPU unit tests" ON)
option(CUAJONE_BUILD_LAUNCHER "Build the Windows GUI launcher" ON)
option(CUAJONE_BUILD_PYTHON_BINDINGS "Build the development/QA pybind11 module" OFF)
option(CUAJONE_ENABLE_TENSORRT "Enable the optional TensorRT/CUDA backend" ON)
option(CUAJONE_ENABLE_TRT_INTEGRATION_TESTS "Build opt-in TensorRT integration tests" OFF)
option(CUAJONE_ENABLE_ONNX_CUDA_INTEGRATION_TESTS "Build opt-in real ONNX Runtime CUDA integration tests" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(WIN32 AND (CUAJONE_BUILD_LAUNCHER OR CUAJONE_BUILD_RUNTIME))
    set(CUAJONE_PRODUCT_ICON "${CMAKE_CURRENT_BINARY_DIR}/CuajonePPEMonitor.ico")
    set(CUAJONE_PRODUCT_RESOURCE "${CMAKE_CURRENT_BINARY_DIR}/product_icon.rc")
    add_custom_command(
        OUTPUT "${CUAJONE_PRODUCT_ICON}"
        COMMAND powershell -NoProfile -ExecutionPolicy Bypass -File
            "${CMAKE_CURRENT_SOURCE_DIR}/../installer/native/generate-icon.ps1"
            -OutputPath "${CUAJONE_PRODUCT_ICON}"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../installer/native/generate-icon.ps1"
        VERBATIM)
    configure_file(resources/launcher.rc.in "${CUAJONE_PRODUCT_RESOURCE}" @ONLY)
    add_custom_target(cuajone_product_icon DEPENDS "${CUAJONE_PRODUCT_ICON}")
endif()

if(CUAJONE_BUILD_LAUNCHER)
    add_library(cuajone_launcher_support
        src/launcher/launcher_support.cpp
    )
    target_include_directories(cuajone_launcher_support PUBLIC include)
    target_compile_features(cuajone_launcher_support PUBLIC cxx_std_20)
    if(MSVC)
        target_compile_options(cuajone_launcher_support PRIVATE /W4 /permissive- /EHsc /utf-8)
    else()
        target_compile_options(cuajone_launcher_support PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    if(WIN32)
        add_executable(cuajone_launcher WIN32 src/launcher/launcher.cpp "${CUAJONE_PRODUCT_RESOURCE}")
        add_dependencies(cuajone_launcher cuajone_product_icon)
        target_link_libraries(cuajone_launcher PRIVATE
            cuajone_launcher_support advapi32 comctl32 ole32 shell32)
        target_compile_definitions(cuajone_launcher PRIVATE
            UNICODE _UNICODE NOMINMAX WIN32_LEAN_AND_MEAN)
        if(MSVC)
            target_compile_options(cuajone_launcher PRIVATE /W4 /permissive- /EHsc /utf-8)
        endif()
    endif()
endif()

if(CUAJONE_BUILD_RUNTIME AND CUAJONE_ENABLE_TENSORRT)
    set(TENSORRT_ROOT "" CACHE PATH "TensorRT 11 C++ SDK root containing include/ and lib/")
    if(NOT TENSORRT_ROOT OR TENSORRT_ROOT STREQUAL "")
        message(FATAL_ERROR
            "TENSORRT_ROOT is required and must point to a TensorRT 11 SDK root")
    endif()
endif()

find_package(OpenCV 4.8 REQUIRED COMPONENTS core imgproc)

set(CUAJONE_BYTE_TRACK_COMMIT "a865158906f6138465668810a98ffd918d95f9a3")
set(CUAJONE_EIGEN_COMMIT "3147391d946bb4b6c68edd901f2add6ac1f31f8c")
set(CUAJONE_BYTE_TRACK_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/../.tools/native/dependencies/byte-track-eigen-${CUAJONE_BYTE_TRACK_COMMIT}"
    CACHE PATH "Verified patched ByteTrack-Eigen 2.1.0 source root")
set(CUAJONE_EIGEN_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/../.tools/native/dependencies/eigen-${CUAJONE_EIGEN_COMMIT}"
    CACHE PATH "Verified Eigen 3.4.0 source root")
if(NOT EXISTS "${CUAJONE_BYTE_TRACK_ROOT}/include/BYTETracker.h"
    OR NOT EXISTS "${CUAJONE_BYTE_TRACK_ROOT}/.cuajone-source-receipt.json"
    OR NOT EXISTS "${CUAJONE_EIGEN_ROOT}/Eigen/Dense"
    OR NOT EXISTS "${CUAJONE_EIGEN_ROOT}/.cuajone-source-receipt.json")
    message(FATAL_ERROR
        "Pinned ByteTrack-Eigen/Eigen sources are missing. Run native/Provision-TrackingDependencies.ps1 first")
endif()
file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/third_party/byte-track-eigen-cuajone.patch"
    CUAJONE_BYTE_TRACK_PATCH_SHA256)
function(cuajone_validate_source_receipt receipt_path expected_archive_sha256)
    file(READ "${receipt_path}" receipt_json)
    string(JSON archive_sha256 GET "${receipt_json}" archive_sha256)
    string(JSON patch_sha256 GET "${receipt_json}" patch_sha256)
    if(NOT archive_sha256 STREQUAL expected_archive_sha256
        OR NOT patch_sha256 STREQUAL CUAJONE_BYTE_TRACK_PATCH_SHA256)
        message(FATAL_ERROR
            "Native tracking dependency receipt does not match approved source and patch hashes: ${receipt_path}")
    endif()
endfunction()
cuajone_validate_source_receipt(
    "${CUAJONE_BYTE_TRACK_ROOT}/.cuajone-source-receipt.json"
    "e5a075df5e8b4ed4bb7436ffe7fe0f4cee5c6a6663112d6a1c47a99ffb704d88")
cuajone_validate_source_receipt(
    "${CUAJONE_EIGEN_ROOT}/.cuajone-source-receipt.json"
    "9eec4ec4e5e459b2f59dbbaa4280e1bb3ee61cccd8a7c0af0321d29d95fece9e")

add_library(cuajone_byte_track_upstream STATIC
    "${CUAJONE_BYTE_TRACK_ROOT}/src/BaseTrack.cpp"
    "${CUAJONE_BYTE_TRACK_ROOT}/src/BoundingBoxIoUMatching.cpp"
    "${CUAJONE_BYTE_TRACK_ROOT}/src/BoundingBoxTrackUtils.cpp"
    "${CUAJONE_BYTE_TRACK_ROOT}/src/BYTETracker.cpp"
    "${CUAJONE_BYTE_TRACK_ROOT}/src/HungarianAlgorithmEigen.cpp"
    "${CUAJONE_BYTE_TRACK_ROOT}/src/KalmanBBoxTrack.cpp"
    "${CUAJONE_BYTE_TRACK_ROOT}/src/KalmanFilter.cpp"
    "${CUAJONE_BYTE_TRACK_ROOT}/src/LinearAssignment.cpp"
)
target_include_directories(cuajone_byte_track_upstream SYSTEM PUBLIC
    "${CUAJONE_BYTE_TRACK_ROOT}/include"
    "${CUAJONE_EIGEN_ROOT}")
target_compile_definitions(cuajone_byte_track_upstream
    PUBLIC EIGEN_MPL2_ONLY BYTE_TRACK_EIGEN_STATIC)
target_compile_features(cuajone_byte_track_upstream PUBLIC cxx_std_20)

add_library(cuajone_byte_track_adapter STATIC src/analytics/byte_tracker.cpp)
target_include_directories(cuajone_byte_track_adapter PUBLIC include)
target_link_libraries(cuajone_byte_track_adapter PRIVATE cuajone_byte_track_upstream)
target_compile_features(cuajone_byte_track_adapter PUBLIC cxx_std_20)

add_library(cuajone_compute
    src/foundation/compute.cpp
)
target_include_directories(cuajone_compute PUBLIC include)
target_compile_features(cuajone_compute PUBLIC cxx_std_20)
if(WIN32)
    target_link_libraries(cuajone_compute PRIVATE dxgi advapi32)
endif()

add_library(cuajone_contracts
    src/foundation/contracts.cpp
)
target_include_directories(cuajone_contracts PUBLIC include)
target_compile_features(cuajone_contracts PUBLIC cxx_std_20)

add_library(cuajone_analytics
    src/analytics/analytics_pipeline.cpp
    src/analytics/fall_analytics.cpp
    src/analytics/letterbox.cpp
    src/analytics/ppe_analytics.cpp
    src/foundation/types.cpp
)
target_include_directories(cuajone_analytics PUBLIC include)
target_link_libraries(cuajone_analytics
    PUBLIC cuajone_contracts
    PRIVATE cuajone_byte_track_adapter)
target_compile_features(cuajone_analytics PUBLIC cxx_std_20)

add_library(cuajone_inference
    src/inference/engine_reader.cpp
    src/inference/preprocess.cpp
    src/inference/yolo_decode.cpp
)
target_include_directories(cuajone_inference PUBLIC include)
target_link_libraries(cuajone_inference PUBLIC cuajone_analytics opencv_core opencv_imgproc)
target_compile_features(cuajone_inference PUBLIC cxx_std_20)

add_library(cuajone_core
    src/app/cli.cpp
    src/app/runtime_execution_plan.cpp
)
target_include_directories(cuajone_core PUBLIC include)
target_link_libraries(cuajone_core PUBLIC cuajone_contracts cuajone_analytics cuajone_inference cuajone_compute)
target_compile_features(cuajone_core PUBLIC cxx_std_20)
if(MSVC)
    foreach(target cuajone_compute cuajone_contracts cuajone_byte_track_adapter cuajone_analytics cuajone_inference cuajone_core)
        target_compile_options(${target} PRIVATE /W4 /permissive- /EHsc)
    endforeach()
else()
    foreach(target cuajone_compute cuajone_contracts cuajone_byte_track_adapter cuajone_analytics cuajone_inference cuajone_core)
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
    endforeach()
endif()

if(CUAJONE_BUILD_PYTHON_BINDINGS)
    set(CUAJONE_PYBIND11_ROOT "" CACHE PATH
        "Pinned pybind11 3.0.4 CMake package root from a local Python environment")
    set(CUAJONE_PYTHON_OUTPUT_DIR "${CMAKE_BINARY_DIR}/python" CACHE PATH
        "Output directory for the development/QA Python extension")
    if(NOT CUAJONE_PYBIND11_ROOT)
        message(FATAL_ERROR
            "CUAJONE_PYBIND11_ROOT is required for bindings; use the pinned pybind11 3.0.4 package for Python 3.12")
    endif()
    find_package(Python 3.12 EXACT COMPONENTS Interpreter Development.Module REQUIRED)
    set(PYBIND11_FINDPYTHON ON)
    find_package(pybind11 3.0.4 EXACT CONFIG REQUIRED
        PATHS "${CUAJONE_PYBIND11_ROOT}" NO_DEFAULT_PATH)
    pybind11_add_module(cuajone_python MODULE src/bindings/python_bindings.cpp)
    target_link_libraries(cuajone_python PRIVATE cuajone_analytics)
    target_compile_features(cuajone_python PRIVATE cxx_std_20)
    set_target_properties(cuajone_python PROPERTIES
        OUTPUT_NAME cuajone_native
        LIBRARY_OUTPUT_DIRECTORY "${CUAJONE_PYTHON_OUTPUT_DIR}"
        RUNTIME_OUTPUT_DIRECTORY "${CUAJONE_PYTHON_OUTPUT_DIR}")
    if(MSVC)
        target_compile_options(cuajone_python PRIVATE /W4 /permissive- /EHsc)
    endif()
endif()

if(CUAJONE_BUILD_TESTS)
    enable_testing()
    add_executable(cuajone_cpu_tests tests/cpu_tests.cpp)
    target_link_libraries(cuajone_cpu_tests PRIVATE cuajone_core)
    add_test(NAME cuajone_cpu_tests COMMAND cuajone_cpu_tests)
    if(WIN32)
        set_tests_properties(cuajone_cpu_tests PROPERTIES
            ENVIRONMENT_MODIFICATION
                "PATH=path_list_prepend:$<TARGET_FILE_DIR:opencv_core>")
    endif()
    if(CUAJONE_BUILD_LAUNCHER)
        add_executable(cuajone_launcher_tests tests/launcher_tests.cpp)
        target_link_libraries(cuajone_launcher_tests PRIVATE cuajone_launcher_support)
        if(MSVC)
            target_compile_options(cuajone_launcher_tests PRIVATE /W4 /permissive- /EHsc)
        endif()
        add_test(NAME cuajone_launcher_tests COMMAND cuajone_launcher_tests)
    endif()
endif()

if(CUAJONE_BUILD_RUNTIME)
    if(NOT WIN32 OR NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
        message(FATAL_ERROR "The native runtime currently supports Windows x64 only")
    endif()
    if(NOT MSVC)
        message(FATAL_ERROR "The native runtime requires the MSVC C++ toolchain")
    endif()

    set(ONNXRUNTIME_ROOT "" CACHE PATH "ONNX Runtime 1.25.0 Windows x64 CPU package root")
    if(NOT ONNXRUNTIME_ROOT)
        message(FATAL_ERROR "ONNXRUNTIME_ROOT is required for the CPU inference backend")
    endif()
    get_filename_component(ONNXRUNTIME_ROOT "${ONNXRUNTIME_ROOT}" ABSOLUTE)
    if(NOT EXISTS "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h"
        OR NOT EXISTS "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib"
        OR NOT EXISTS "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll")
        message(FATAL_ERROR
            "ONNXRUNTIME_ROOT must contain ONNX Runtime 1.25.0 include/ and lib/ files")
    endif()
    add_library(cuajone_onnxruntime SHARED IMPORTED)
    set_target_properties(cuajone_onnxruntime PROPERTIES
        IMPORTED_IMPLIB "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib"
        IMPORTED_LOCATION "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
        INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_ROOT}/include")

    if(CUAJONE_ENABLE_TENSORRT)
        get_filename_component(TENSORRT_ROOT "${TENSORRT_ROOT}" ABSOLUTE)
        unset(TENSORRT_INCLUDE_DIR CACHE)
        unset(TENSORRT_NVINFER_LIBRARY CACHE)
        find_path(TENSORRT_INCLUDE_DIR NvInfer.h PATHS "${TENSORRT_ROOT}/include" NO_DEFAULT_PATH)
        find_library(TENSORRT_NVINFER_LIBRARY NAMES nvinfer_11
            PATHS "${TENSORRT_ROOT}/lib" "${TENSORRT_ROOT}/lib/x64" NO_DEFAULT_PATH)
        if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_NVINFER_LIBRARY)
            message(FATAL_ERROR "TensorRT 11 headers and nvinfer_11 import library are required")
        endif()
        set(CUDA_RUNTIME_ROOT "" CACHE PATH
            "Optional CUDA runtime-only root containing include/, lib/x64/, and bin/")
        set(CUDA_COMPILER_HEADERS_ROOT "" CACHE PATH "Optional split CUDA compiler headers root")
        set(CUDA_CCCL_HEADERS_ROOT "" CACHE PATH "Optional split CUDA CCCL headers root")
        if(CUDA_RUNTIME_ROOT)
            get_filename_component(CUDA_RUNTIME_ROOT "${CUDA_RUNTIME_ROOT}" ABSOLUTE)
            find_path(CUDA_RUNTIME_INCLUDE_DIR cuda_runtime_api.h PATHS "${CUDA_RUNTIME_ROOT}/include" NO_DEFAULT_PATH)
            find_library(CUDA_CUDART_LIBRARY NAMES cudart PATHS "${CUDA_RUNTIME_ROOT}/lib/x64" NO_DEFAULT_PATH)
            find_file(CUDA_CUDART_DLL NAMES cudart64_12.dll PATHS "${CUDA_RUNTIME_ROOT}/bin" NO_DEFAULT_PATH)
            if(NOT CUDA_RUNTIME_INCLUDE_DIR OR NOT CUDA_CUDART_LIBRARY OR NOT CUDA_CUDART_DLL)
                message(FATAL_ERROR "CUDA_RUNTIME_ROOT is incomplete")
            endif()
            set(CUDA_RUNTIME_INCLUDE_DIRS "${CUDA_RUNTIME_INCLUDE_DIR}")
            if(NOT EXISTS "${CUDA_RUNTIME_INCLUDE_DIR}/crt/host_defines.h")
                if(NOT EXISTS "${CUDA_COMPILER_HEADERS_ROOT}/include/crt/host_defines.h")
                    message(FATAL_ERROR "CUDA_COMPILER_HEADERS_ROOT is required for split runtime headers")
                endif()
                list(APPEND CUDA_RUNTIME_INCLUDE_DIRS "${CUDA_COMPILER_HEADERS_ROOT}/include")
            endif()
            if(NOT EXISTS "${CUDA_RUNTIME_INCLUDE_DIR}/nv/target"
                AND NOT EXISTS "${CUDA_COMPILER_HEADERS_ROOT}/include/nv/target")
                if(NOT EXISTS "${CUDA_CCCL_HEADERS_ROOT}/include/nv/target")
                    message(FATAL_ERROR "CUDA_CCCL_HEADERS_ROOT is required for split runtime headers")
                endif()
                list(APPEND CUDA_RUNTIME_INCLUDE_DIRS "${CUDA_CCCL_HEADERS_ROOT}/include")
            endif()
            add_library(cuajone_cuda_runtime SHARED IMPORTED)
            set_target_properties(cuajone_cuda_runtime PROPERTIES
                IMPORTED_IMPLIB "${CUDA_CUDART_LIBRARY}"
                IMPORTED_LOCATION "${CUDA_CUDART_DLL}"
                INTERFACE_INCLUDE_DIRECTORIES "${CUDA_RUNTIME_INCLUDE_DIRS}")
            set(CUAJONE_CUDART_TARGET cuajone_cuda_runtime)
        else()
            find_package(CUDAToolkit 12 REQUIRED)
            set(CUAJONE_CUDART_TARGET CUDA::cudart)
        endif()
    endif()
    find_package(OpenCV 4.8 REQUIRED COMPONENTS core imgproc imgcodecs videoio highgui)

    add_library(cuajone_runtime
        src/runtime/capture.cpp
        src/runtime/engine_pipeline.cpp
        src/runtime/evidence.cpp
        src/runtime/model_manifest.cpp
        src/runtime/onnx_session.cpp
    )
    target_include_directories(cuajone_runtime PUBLIC include)
    target_link_libraries(cuajone_runtime
        PUBLIC cuajone_core
        PRIVATE cuajone_onnxruntime
            bcrypt
            opencv_imgcodecs opencv_videoio opencv_highgui)
    if(CUAJONE_ENABLE_TENSORRT)
        target_sources(cuajone_runtime PRIVATE src/runtime/tensorrt_runtime.cpp)
        target_include_directories(cuajone_runtime PRIVATE "${TENSORRT_INCLUDE_DIR}")
        target_link_libraries(cuajone_runtime PUBLIC ${CUAJONE_CUDART_TARGET})
        target_link_libraries(cuajone_runtime PRIVATE "${TENSORRT_NVINFER_LIBRARY}")
        target_compile_definitions(cuajone_runtime PRIVATE CUAJONE_WITH_TENSORRT=1)
    endif()
    target_compile_options(cuajone_runtime PRIVATE /W4 /permissive- /EHsc)

    if(TARGET cuajone_python)
        target_compile_definitions(cuajone_python PRIVATE CUAJONE_PYTHON_WITH_ENGINE_RUNTIME=1)
        target_link_libraries(cuajone_python PRIVATE cuajone_runtime)
        add_custom_command(TARGET cuajone_python POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:cuajone_onnxruntime>
                $<TARGET_FILE_DIR:cuajone_python>)
        if(CUAJONE_ENABLE_TENSORRT)
            target_link_libraries(cuajone_python PRIVATE delayimp)
            target_link_options(cuajone_python PRIVATE
                "/DELAYLOAD:cudart64_12.dll"
                "/DELAYLOAD:nvinfer_11.dll")
        endif()
    endif()

    add_executable(cuajone_native src/app/main.cpp "${CUAJONE_PRODUCT_RESOURCE}")
    add_dependencies(cuajone_native cuajone_product_icon)
    target_link_libraries(cuajone_native PRIVATE cuajone_runtime)
    target_compile_options(cuajone_native PRIVATE /W4 /permissive- /EHsc /utf-8)
    add_custom_command(TARGET cuajone_native POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:cuajone_onnxruntime>
            $<TARGET_FILE_DIR:cuajone_native>)
    if(CUAJONE_ENABLE_TENSORRT)
        target_link_libraries(cuajone_native PRIVATE delayimp)
        target_link_options(cuajone_native PRIVATE
            "/DELAYLOAD:cudart64_12.dll"
            "/DELAYLOAD:nvinfer_11.dll")
    endif()

    add_library(cuajone_installer_ca SHARED
        src/foundation/compute.cpp
        src/installer/installer_custom_action.cpp
    )
    target_include_directories(cuajone_installer_ca PRIVATE include)
    target_link_libraries(cuajone_installer_ca PRIVATE dxgi advapi32 msi)
    target_compile_options(cuajone_installer_ca PRIVATE /W4 /permissive- /EHsc)
    set_target_properties(cuajone_installer_ca PROPERTIES
        OUTPUT_NAME CuajoneHardwareProbeCA
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

    if(CUAJONE_BUILD_TESTS)
        add_executable(cuajone_evidence_tests tests/evidence_tests.cpp)
        target_link_libraries(cuajone_evidence_tests PRIVATE cuajone_runtime)
        target_compile_options(cuajone_evidence_tests PRIVATE /W4 /permissive- /EHsc)
        add_test(NAME cuajone_evidence_tests COMMAND cuajone_evidence_tests)
        set_tests_properties(cuajone_evidence_tests PROPERTIES
            ENVIRONMENT_MODIFICATION
                "PATH=path_list_prepend:$<TARGET_FILE_DIR:opencv_core>;PATH=path_list_prepend:${ONNXRUNTIME_ROOT}/lib")

        add_executable(cuajone_onnx_tests tests/onnx_tests.cpp)
        target_link_libraries(cuajone_onnx_tests PRIVATE cuajone_runtime)
        target_compile_options(cuajone_onnx_tests PRIVATE /W4 /permissive- /EHsc)
        add_custom_command(TARGET cuajone_onnx_tests POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:cuajone_onnxruntime>
                $<TARGET_FILE_DIR:cuajone_onnx_tests>)
        add_test(NAME cuajone_onnx_tests COMMAND cuajone_onnx_tests)
        set_tests_properties(cuajone_onnx_tests PROPERTIES
            ENVIRONMENT_MODIFICATION
                "PATH=path_list_prepend:$<TARGET_FILE_DIR:opencv_core>;PATH=path_list_prepend:${ONNXRUNTIME_ROOT}/lib")
    endif()

    if(CUAJONE_ENABLE_TRT_INTEGRATION_TESTS AND CUAJONE_ENABLE_TENSORRT)
        set(CUAJONE_TEST_ENGINE "" CACHE FILEPATH "Engine used by the opt-in TensorRT smoke test")
        if(NOT EXISTS "${CUAJONE_TEST_ENGINE}")
            message(FATAL_ERROR "CUAJONE_TEST_ENGINE must point to a compatible engine")
        endif()
        add_executable(cuajone_trt_smoke tests/trt_smoke.cpp)
        target_link_libraries(cuajone_trt_smoke PRIVATE cuajone_runtime)
        add_test(NAME cuajone_trt_smoke COMMAND cuajone_trt_smoke "${CUAJONE_TEST_ENGINE}")
    endif()

    if(CUAJONE_ENABLE_ONNX_CUDA_INTEGRATION_TESTS)
        set(CUAJONE_ONNX_CUDA_RUNTIME_DIR "" CACHE PATH
            "Directory containing the staged ONNX Runtime CUDA provider DLL closure")
        set(CUAJONE_ONNX_CUDA_PPE_MODEL "" CACHE FILEPATH
            "Verified PPE ONNX model used by the opt-in CUDA pipeline integration test")
        set(CUAJONE_ONNX_CUDA_POSE_MODEL "" CACHE FILEPATH
            "Verified pose ONNX model used by the opt-in CUDA pipeline integration test")
        set(CUAJONE_ONNX_REAL_PERSON_IMAGE
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/person_regression.jpg"
            CACHE FILEPATH "Offline person image used by the native real-model regression")
        foreach(required_file
                "${CUAJONE_ONNX_CUDA_RUNTIME_DIR}/onnxruntime_providers_cuda.dll"
                "${CUAJONE_ONNX_CUDA_RUNTIME_DIR}/onnxruntime_providers_shared.dll"
                "${CUAJONE_ONNX_CUDA_PPE_MODEL}"
                "${CUAJONE_ONNX_CUDA_PPE_MODEL}.manifest.json"
                "${CUAJONE_ONNX_CUDA_POSE_MODEL}"
                "${CUAJONE_ONNX_CUDA_POSE_MODEL}.manifest.json"
                "${CUAJONE_ONNX_REAL_PERSON_IMAGE}")
            if(NOT EXISTS "${required_file}")
                message(FATAL_ERROR
                    "ONNX CUDA integration prerequisite is missing: ${required_file}")
            endif()
        endforeach()
        add_executable(cuajone_onnx_cuda_integration_tests tests/onnx_cuda_integration_tests.cpp)
        target_link_libraries(cuajone_onnx_cuda_integration_tests PRIVATE cuajone_runtime cuajone_onnxruntime)
        target_compile_options(cuajone_onnx_cuda_integration_tests PRIVATE /W4 /permissive- /EHsc)
        add_custom_command(TARGET cuajone_onnx_cuda_integration_tests POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:cuajone_onnxruntime>
                $<TARGET_FILE_DIR:cuajone_onnx_cuda_integration_tests>
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CUAJONE_ONNX_CUDA_RUNTIME_DIR}/onnxruntime_providers_cuda.dll"
                $<TARGET_FILE_DIR:cuajone_onnx_cuda_integration_tests>
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CUAJONE_ONNX_CUDA_RUNTIME_DIR}/onnxruntime_providers_shared.dll"
                $<TARGET_FILE_DIR:cuajone_onnx_cuda_integration_tests>)
        add_test(NAME cuajone_onnx_pose_cpu_standalone_tests
            COMMAND cuajone_onnx_cuda_integration_tests
                standalone-pose-cpu "${CUAJONE_ONNX_CUDA_POSE_MODEL}")
        add_test(NAME cuajone_onnx_cuda_ppe_only_tests
            COMMAND cuajone_onnx_cuda_integration_tests
                ppe-only "${CUAJONE_ONNX_CUDA_PPE_MODEL}")
        add_test(NAME cuajone_onnx_cuda_hybrid_pipeline_tests
            COMMAND cuajone_onnx_cuda_integration_tests pipeline
                "${CUAJONE_ONNX_CUDA_PPE_MODEL}" "${CUAJONE_ONNX_CUDA_POSE_MODEL}"
                "${CUAJONE_ONNX_REAL_PERSON_IMAGE}")
        set_tests_properties(
            cuajone_onnx_pose_cpu_standalone_tests
            cuajone_onnx_cuda_ppe_only_tests
            cuajone_onnx_cuda_hybrid_pipeline_tests
            PROPERTIES
            ENVIRONMENT "CUAJONE_ONNX_CUDA_RUNTIME_DIR=${CUAJONE_ONNX_CUDA_RUNTIME_DIR}"
            ENVIRONMENT_MODIFICATION
                "PATH=path_list_prepend:$<TARGET_FILE_DIR:opencv_core>;PATH=path_list_prepend:${ONNXRUNTIME_ROOT}/lib")
        set_tests_properties(cuajone_onnx_pose_cpu_standalone_tests PROPERTIES
            LABELS "onnx;acceptance;pose-cpu;hybrid-rationale-ort-cuda-1.25-gtx1650ti")
        set_tests_properties(
            cuajone_onnx_cuda_ppe_only_tests
            cuajone_onnx_cuda_hybrid_pipeline_tests
            PROPERTIES
            SKIP_RETURN_CODE 77)
        set_tests_properties(cuajone_onnx_cuda_ppe_only_tests PROPERTIES
            LABELS "gpu;onnx;acceptance;ppe-cuda")
        set_tests_properties(cuajone_onnx_cuda_hybrid_pipeline_tests PROPERTIES
            LABELS "gpu;onnx;acceptance;hybrid;ppe-cuda;pose-cpu;known-pose-cuda-access-violation")
    endif()
endif()
