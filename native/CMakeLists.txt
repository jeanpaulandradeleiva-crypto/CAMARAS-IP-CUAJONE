# SPDX-License-Identifier: AGPL-3.0-only

cmake_minimum_required(VERSION 3.25)

project(cuajone_native LANGUAGES CXX)

option(CUAJONE_BUILD_RUNTIME "Build the Windows TensorRT runtime" ON)
option(CUAJONE_BUILD_TESTS "Build pure CPU unit tests" ON)
option(CUAJONE_ENABLE_TRT_INTEGRATION_TESTS "Build opt-in TensorRT integration tests" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CUAJONE_BUILD_RUNTIME)
    set(TENSORRT_ROOT "" CACHE PATH "TensorRT 11 C++ SDK root containing include/ and lib/")
    if(NOT TENSORRT_ROOT OR TENSORRT_ROOT STREQUAL "")
        message(FATAL_ERROR
            "TENSORRT_ROOT is required and must point to a TensorRT 11 SDK root")
    endif()
endif()

find_package(OpenCV 4.8 REQUIRED COMPONENTS core imgproc)

add_library(cuajone_core
    src/cli.cpp
    src/engine_reader.cpp
    src/fall_analytics.cpp
    src/iou_tracker.cpp
    src/letterbox.cpp
    src/ppe_analytics.cpp
    src/preprocess.cpp
    src/types.cpp
    src/yolo_decode.cpp
)
target_include_directories(cuajone_core PUBLIC include)
target_link_libraries(cuajone_core PUBLIC opencv_core opencv_imgproc)
target_compile_features(cuajone_core PUBLIC cxx_std_20)
if(MSVC)
    target_compile_options(cuajone_core PRIVATE /W4 /permissive- /EHsc)
else()
    target_compile_options(cuajone_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(CUAJONE_BUILD_TESTS)
    enable_testing()
    add_executable(cuajone_cpu_tests tests/cpu_tests.cpp)
    target_link_libraries(cuajone_cpu_tests PRIVATE cuajone_core)
    add_test(NAME cuajone_cpu_tests COMMAND cuajone_cpu_tests)
endif()

if(CUAJONE_BUILD_RUNTIME)
    if(NOT WIN32 OR NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
        message(FATAL_ERROR "The native runtime currently supports Windows x64 only")
    endif()
    if(NOT MSVC)
        message(FATAL_ERROR "The native runtime requires the MSVC C++ toolchain")
    endif()

    get_filename_component(TENSORRT_ROOT "${TENSORRT_ROOT}" ABSOLUTE)
    unset(TENSORRT_INCLUDE_DIR CACHE)
    unset(TENSORRT_NVINFER_LIBRARY CACHE)
    find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        PATHS "${TENSORRT_ROOT}/include"
        NO_DEFAULT_PATH)
    if(NOT TENSORRT_INCLUDE_DIR)
        message(FATAL_ERROR
            "TensorRT 11 header NvInfer.h was not found under ${TENSORRT_ROOT}/include")
    endif()
    find_library(TENSORRT_NVINFER_LIBRARY NAMES nvinfer_11
        PATHS "${TENSORRT_ROOT}/lib" "${TENSORRT_ROOT}/lib/x64"
        NO_DEFAULT_PATH)
    if(NOT TENSORRT_NVINFER_LIBRARY)
        message(FATAL_ERROR
            "TensorRT 11 import library nvinfer_11 was not found under "
            "${TENSORRT_ROOT}/lib or ${TENSORRT_ROOT}/lib/x64")
    endif()
    message(STATUS "TensorRT expectation: major version 11, nvinfer_11 only")
    message(STATUS "TensorRT header: ${TENSORRT_INCLUDE_DIR}/NvInfer.h")
    message(STATUS "TensorRT library: ${TENSORRT_NVINFER_LIBRARY}")

    set(CUDA_RUNTIME_ROOT "" CACHE PATH
        "Optional CUDA runtime-only root containing include/, lib/x64/, and bin/")
    set(CUDA_COMPILER_HEADERS_ROOT "" CACHE PATH
        "Optional split CUDA package root containing include/crt/")
    set(CUDA_CCCL_HEADERS_ROOT "" CACHE PATH
        "Optional split CUDA CCCL package root containing include/nv/target")
    if(CUDA_RUNTIME_ROOT)
        get_filename_component(CUDA_RUNTIME_ROOT "${CUDA_RUNTIME_ROOT}" ABSOLUTE)
        unset(CUDA_RUNTIME_INCLUDE_DIR CACHE)
        unset(CUDA_CUDART_LIBRARY CACHE)
        unset(CUDA_CUDART_DLL CACHE)
        find_path(CUDA_RUNTIME_INCLUDE_DIR cuda_runtime_api.h
            PATHS "${CUDA_RUNTIME_ROOT}/include"
            NO_DEFAULT_PATH)
        find_library(CUDA_CUDART_LIBRARY NAMES cudart
            PATHS "${CUDA_RUNTIME_ROOT}/lib/x64" "${CUDA_RUNTIME_ROOT}/lib"
            NO_DEFAULT_PATH)
        find_file(CUDA_CUDART_DLL NAMES cudart64_12.dll
            PATHS "${CUDA_RUNTIME_ROOT}/bin"
            NO_DEFAULT_PATH)
        if(NOT CUDA_RUNTIME_INCLUDE_DIR OR NOT CUDA_CUDART_LIBRARY OR NOT CUDA_CUDART_DLL)
            message(FATAL_ERROR
                "CUDA_RUNTIME_ROOT must contain include/cuda_runtime_api.h, "
                "lib/x64/cudart.lib, and bin/cudart64_12.dll")
        endif()
        set(CUDA_RUNTIME_INCLUDE_DIRS "${CUDA_RUNTIME_INCLUDE_DIR}")
        if(NOT EXISTS "${CUDA_RUNTIME_INCLUDE_DIR}/crt/host_defines.h")
            get_filename_component(CUDA_COMPILER_HEADERS_ROOT
                "${CUDA_COMPILER_HEADERS_ROOT}" ABSOLUTE)
            if(NOT EXISTS "${CUDA_COMPILER_HEADERS_ROOT}/include/crt/host_defines.h")
                message(FATAL_ERROR
                    "CUDA runtime headers reference crt/host_defines.h; set "
                    "CUDA_COMPILER_HEADERS_ROOT to the matching split NVIDIA package root")
            endif()
            list(APPEND CUDA_RUNTIME_INCLUDE_DIRS
                "${CUDA_COMPILER_HEADERS_ROOT}/include")
            message(STATUS "CUDA compiler headers: ${CUDA_COMPILER_HEADERS_ROOT}/include")
        endif()
        if(NOT EXISTS "${CUDA_RUNTIME_INCLUDE_DIR}/nv/target"
                AND NOT EXISTS "${CUDA_COMPILER_HEADERS_ROOT}/include/nv/target")
            get_filename_component(CUDA_CCCL_HEADERS_ROOT
                "${CUDA_CCCL_HEADERS_ROOT}" ABSOLUTE)
            if(NOT EXISTS "${CUDA_CCCL_HEADERS_ROOT}/include/nv/target")
                message(FATAL_ERROR
                    "CUDA runtime headers reference nv/target; set "
                    "CUDA_CCCL_HEADERS_ROOT to the matching split NVIDIA package root")
            endif()
            list(APPEND CUDA_RUNTIME_INCLUDE_DIRS "${CUDA_CCCL_HEADERS_ROOT}/include")
            message(STATUS "CUDA CCCL headers: ${CUDA_CCCL_HEADERS_ROOT}/include")
        endif()
        add_library(cuajone_cuda_runtime SHARED IMPORTED)
        set_target_properties(cuajone_cuda_runtime PROPERTIES
            IMPORTED_IMPLIB "${CUDA_CUDART_LIBRARY}"
            IMPORTED_LOCATION "${CUDA_CUDART_DLL}"
            INTERFACE_INCLUDE_DIRECTORIES "${CUDA_RUNTIME_INCLUDE_DIRS}")
        set(CUAJONE_CUDART_TARGET cuajone_cuda_runtime)
        message(STATUS "CUDA runtime-only root: ${CUDA_RUNTIME_ROOT}")
        message(STATUS "CUDA runtime library: ${CUDA_CUDART_LIBRARY}")
    else()
        find_package(CUDAToolkit 12 REQUIRED)
        set(CUAJONE_CUDART_TARGET CUDA::cudart)
    endif()
    find_package(OpenCV 4.8 REQUIRED COMPONENTS core imgproc imgcodecs videoio highgui)

    add_library(cuajone_runtime
        src/capture.cpp
        src/evidence.cpp
        src/tensorrt_runtime.cpp
    )
    target_include_directories(cuajone_runtime PUBLIC include "${TENSORRT_INCLUDE_DIR}")
    target_link_libraries(cuajone_runtime
        PUBLIC cuajone_core
        PUBLIC ${CUAJONE_CUDART_TARGET}
        PRIVATE "${TENSORRT_NVINFER_LIBRARY}"
            opencv_imgcodecs opencv_videoio opencv_highgui)
    target_compile_options(cuajone_runtime PRIVATE /W4 /permissive- /EHsc)

    add_executable(cuajone_native src/main.cpp)
    target_link_libraries(cuajone_native PRIVATE cuajone_runtime)

    if(CUAJONE_ENABLE_TRT_INTEGRATION_TESTS)
        set(CUAJONE_TEST_ENGINE "" CACHE FILEPATH "Engine used by the opt-in TensorRT smoke test")
        if(NOT EXISTS "${CUAJONE_TEST_ENGINE}")
            message(FATAL_ERROR "CUAJONE_TEST_ENGINE must point to a compatible engine")
        endif()
        add_executable(cuajone_trt_smoke tests/trt_smoke.cpp)
        target_link_libraries(cuajone_trt_smoke PRIVATE cuajone_runtime)
        add_test(NAME cuajone_trt_smoke COMMAND cuajone_trt_smoke "${CUAJONE_TEST_ENGINE}")
    endif()
endif()
