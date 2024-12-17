# FindTensorRT.cmake

# 设置 TensorRT 的根目录（可根据需要修改）
if (NOT TENSORRT_ROOT_DIR)
    set(TENSORRT_ROOT_DIR "/usr/local/TensorRT")
endif()

find_path(TENSORRT_INCLUDE_DIR
    NAMES NvInfer.h
    PATHS ${TENSORRT_ROOT_DIR}
    PATH_SUFFIXES include
)

find_library(TENSORRT_NVINFER_LIBRARY
    NAMES nvinfer
    PATHS ${TENSORRT_ROOT_DIR}
    PATH_SUFFIXES lib
)

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

if (TENSORRT_INCLUDE_DIR)
    # 查找头文件
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
    set(TensorRT_VERSION ${TENSORRT_MAJOR}.${TENSORRT_MINOR}.${TENSORRT_PATCH}.${TENSORRT_BUILD})
    if (NOT TensorRT_VERSION)
        message(FATAL_ERROR "Unable to extract version from NvInferVersion.h")
    endif()
endif()


# 设置变量
if(TENSORRT_INCLUDE_DIR AND TENSORRT_NVINFER_LIBRARY)
    set(TensorRT_FOUND TRUE)
    set(TensorRT_INCLUDE_DIR ${TENSORRT_INCLUDE_DIR})
    set(TensorRT_LIBRARIES ${TENSORRT_NVINFER_LIBRARY})
    set(TensorRT_STATIC_LIBRARIES ${TENSORRT_NVINFER_STATIC_LIBRARY})
    set(TensorRT_LIBRARY_DIR ${TENSORRT_ROOT_DIR}/lib)
else()
    set(TensorRT_FOUND FALSE)
endif()

# 输出查找结果
if(TensorRT_FOUND)
    message(STATUS "Found TensorRT: ${TENSORRT_ROOT_DIR} (found version \"${TensorRT_VERSION}\")")
    message(STATUS "Found TensorRT_LIBRARIES: ${TensorRT_LIBRARIES}")
else()
    message(FATAL_ERROR "TensorRT not found")
endif()
