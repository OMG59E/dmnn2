# ==============================================================================
# FindTensorRT.cmake - Find NVIDIA TensorRT library
# ==============================================================================
#
# This module finds TensorRT and sets the following variables:
#
#   TensorRT_FOUND        - True if TensorRT was found
#   TensorRT_INCLUDE_DIR  - TensorRT include directory
#   TensorRT_LIBRARIES    - TensorRT libraries
#   TensorRT_VERSION      - TensorRT version string
#
# ==============================================================================

# Set default TensorRT root directory
if(NOT TENSORRT_ROOT_DIR)
    set(TENSORRT_ROOT_DIR "/usr/local/TensorRT")
endif()

# Find include directory
find_path(TENSORRT_INCLUDE_DIR
    NAMES NvInfer.h
    PATHS ${TENSORRT_ROOT_DIR}
    PATH_SUFFIXES include
)

# Find nvinfer library
find_library(TENSORRT_NVINFER_LIBRARY
    NAMES nvinfer
    PATHS ${TENSORRT_ROOT_DIR}
    PATH_SUFFIXES lib
)

# Find static libraries
find_library(TENSORRT_NVINFER_STATIC_LIBRARY
    NAMES nvinfer_static
    PATHS ${TENSORRT_ROOT_DIR}
    PATH_SUFFIXES lib
)

find_library(TENSORRT_NVINFER_PLUGIN_STATIC_LIBRARY
    NAMES nvinfer_plugin_static
    PATHS ${TENSORRT_ROOT_DIR}
    PATH_SUFFIXES lib
)

# Extract version from header
if(TENSORRT_INCLUDE_DIR)
    find_path(TENSORRT_INCLUDE_DIR NAMES NvInferVersion.h)
    file(READ "${TENSORRT_INCLUDE_DIR}/NvInferVersion.h" NvInferVersionContent)
    string(REGEX MATCH "#define NV_TENSORRT_MAJOR[ \t]+([0-9]+)" _major_match "${NvInferVersionContent}")
    set(TENSORRT_MAJOR ${CMAKE_MATCH_1})
    string(REGEX MATCH "#define NV_TENSORRT_MINOR[ \t]+([0-9]+)" _minor_match "${NvInferVersionContent}")
    set(TENSORRT_MINOR ${CMAKE_MATCH_1})
    string(REGEX MATCH "#define NV_TENSORRT_PATCH[ \t]+([0-9]+)" _patch_match "${NvInferVersionContent}")
    set(TENSORRT_PATCH ${CMAKE_MATCH_1})
    string(REGEX MATCH "#define NV_TENSORRT_BUILD[ \t]+([0-9]+)" _build_match "${NvInferVersionContent}")
    set(TENSORRT_BUILD ${CMAKE_MATCH_1})
    set(TensorRT_VERSION "${TENSORRT_MAJOR}.${TENSORRT_MINOR}.${TENSORRT_PATCH}.${TENSORRT_BUILD}")
endif()

# Set output variables
if(TENSORRT_INCLUDE_DIR AND TENSORRT_NVINFER_LIBRARY)
    set(TensorRT_FOUND TRUE)
    set(TensorRT_INCLUDE_DIR ${TENSORRT_INCLUDE_DIR})
    set(TensorRT_LIBRARIES ${TENSORRT_NVINFER_LIBRARY})
    set(TensorRT_STATIC_LIBRARIES ${TENSORRT_NVINFER_STATIC_LIBRARY})
    set(TensorRT_LIBRARY_DIR ${TENSORRT_ROOT_DIR}/lib)
else()
    set(TensorRT_FOUND FALSE)
endif()

# Handle REQUIRED argument
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorRT
    REQUIRED_VARS TensorRT_LIBRARIES TENSORRT_INCLUDE_DIR
    VERSION_VAR TensorRT_VERSION
)

# Print status
if(TensorRT_FOUND)
    message(STATUS "Found TensorRT: ${TENSORRT_ROOT_DIR} (version ${TensorRT_VERSION})")
    message(STATUS "TensorRT libraries: ${TensorRT_LIBRARIES}")
endif()