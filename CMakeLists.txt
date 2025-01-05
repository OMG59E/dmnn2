cmake_minimum_required(VERSION 3.10)
project(dmnn C CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

cmake_policy(SET CMP0077 NEW)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w -g -fPIC")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w -O2 -fPIC")  # -Wl,-Bsymbolic
endif()
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_NVCC_FLAGS -w; -O2; -Xcompiler -fPIC;
    -gencode arch=compute_61,code=sm_61;  # GTX1060/1070/1080/1080Ti
    -gencode arch=compute_70,code=sm_70;  # GTX V100
    -gencode arch=compute_75,code=sm_75;  # RTX2060/2070/2080/2080Ti, Tesla T4, GTX1660
    -gencode arch=compute_86,code=sm_86;  # RTX3060/3070/3080/3080Ti/3090, A1000/2000/3000/4000
)

# CUDA
cmake_policy(SET CMP0146 OLD)
find_package(CUDA 11.1 EXACT QUIET REQUIRED)
if (CUDA_FOUND)
    message(STATUS "Found CUDA: ${CUDA_TOOLKIT_ROOT_DIR} (found version \"${CUDA_VERSION}\")")
    link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
else()
    message(FATAL_ERROR "Not found CUDA")
endif()
find_library(CUDA_CUDNN_LIBRARIES NAMES cudnn HINTS "${CUDA_TOOLKIT_ROOT_DIR}/lib64")
if(CUDA_CUDNN_LIBRARIES)
    find_path(CUDNN_INCLUDE_DIR NAMES cudnn_version.h PATHS ${CUDA_INCLUDE_DIRS})
    file(READ "${CUDNN_INCLUDE_DIR}/cudnn_version.h" CuDNNVersionContent)
    string(REGEX MATCH "#define CUDNN_MAJOR ([0-9]+)" _major_match "${CuDNNVersionContent}")
    set(CUDNN_VERSION_MAJOR ${CMAKE_MATCH_1})
    string(REGEX MATCH "#define CUDNN_MINOR ([0-9]+)" _minor_match "${CuDNNVersionContent}")
    set(CUDNN_VERSION_MINOR ${CMAKE_MATCH_1})
    string(REGEX MATCH "#define CUDNN_PATCHLEVEL ([0-9]+)" _patch_match "${CuDNNVersionContent}")
    set(CUDNN_VERSION_PATCH ${CMAKE_MATCH_1})
    set(CUDNN_VERSION ${CUDNN_VERSION_MAJOR}.${CUDNN_VERSION_MINOR}.${CUDNN_VERSION_PATCH})
    if (CUDNN_VERSION)
        message(STATUS "Found cuDNN: ${CUDA_CUDNN_LIBRARIES} (found version \"${CUDNN_VERSION}\")")
    else()
        message(FATAL_ERROR "Unable to extract version from cudnn_version.h")
    endif()
else()
    message(FATAL_ERROR "Not found cuDNN")
endif()
# nvJPEG
find_library(NVJPEG_LIBRARIES nvjpeg HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
if(NVJPEG_LIBRARIES)
    message(STATUS "Found nvJPEG: ${NVJPEG_LIBRARIES}")
else()
    message(FATAL_ERROR "Not found nvJPEG")
endif()

# TensorRT
find_package(TensorRT REQUIRED)

# Protobuf
find_package(Protobuf QUIET REQUIRED)
message(STATUS "Found Protobuf: ${Protobuf_LIBRARIES} (found version \"${Protobuf_VERSION}\")")
protobuf_generate_cpp(TRT_CAFFE_PROTO_SRCS TRT_CAFFE_PROTO_HDRS ${PROJECT_SOURCE_DIR}/src/parsers/caffe/proto/trtcaffe.proto)
protobuf_generate_cpp(TRT_ONNX_PROTO_SRCS TRT_ONNX_PROTO_HDRS ${PROJECT_SOURCE_DIR}/src/parsers/onnx/proto/onnx-ml.proto)
message(STATUS "Generated trtcaffe.proto: ${TRT_CAFFE_PROTO_SRCS};${TRT_CAFFE_PROTO_HDRS}")
message(STATUS "Generated onnx-ml.proto: ${TRT_ONNX_PROTO_SRCS};${TRT_ONNX_PROTO_HDRS}")
#
add_subdirectory(tools)
add_subdirectory(src)
add_subdirectory(samples)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${TensorRT_INCLUDE_DIR})
include_directories(include)
include_directories(include/utils/spdlog/include)
include_directories(include/utils/argparse/include)
include_directories(src/utils)
include_directories(src/parsers/onnx/)
# 3rdparty
include_directories(3rdparty/include)
link_directories(3rdparty/lib)
set(FFMPEG_LIBRARIES avcodec avformat avutil swscale swresample)
set(NV_DRIVER_LIBRARIES nvcuvid cuda)
if (EXISTS "/proc/sys/fs/binfmt_misc/WSLInterop")
    message(STATUS "Running on WSL")
    link_directories(/usr/lib/wsl/lib)
endif()

#
cuda_add_library(${PROJECT_NAME} SHARED ${DMNN_SOURCES} ${TRT_CAFFE_PROTO_SRCS} ${TRT_ONNX_PROTO_SRCS})
target_link_libraries(${PROJECT_NAME} ${TensorRT_LIBRARIES} ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} 
    ${CUDA_CUDNN_LIBRARIES} ${Protobuf_LIBRARIES} ${NVJPEG_LIBRARIES} ${FFMPEG_LIBRARIES} ${NV_DRIVER_LIBRARIES})

#
add_subdirectory(python/pybind11)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python)
pybind11_add_module(pydmnn SHARED python/pydmnn.cpp)
target_link_libraries(pydmnn PRIVATE pybind11::module)

macro(add_executables EXECUTABLE_LIST)
    foreach(EXECUTABLE_NAME ${EXECUTABLE_LIST})
        string(REPLACE "/" ";" SEXY_LIST ${EXECUTABLE_NAME})
        list(GET SEXY_LIST -1 TEST_CASE_NAME)
        string(REPLACE "." ";" SEXY_LIST ${TEST_CASE_NAME})
        list(GET SEXY_LIST 0 TEST_CASE_NAME)
        cuda_add_executable(${TEST_CASE_NAME} ${EXECUTABLE_NAME})
        target_link_libraries(${TEST_CASE_NAME} ${PROJECT_NAME} stdc++fs)
    endforeach()
endmacro()

add_executables("${TOOL_SOURCES}")
add_executables("${SAMPLE_SOURCES}")
