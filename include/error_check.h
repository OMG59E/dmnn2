/***
 * @Author: xingwg
 * @Date: 2024-10-10 13:50:36
 * @LastEditTime: 2024-10-31 16:27:44
 * @FilePath: /dmnn2/include/error_check.h
 * @Description:
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#pragma once
#include "logging.h"
#include <NvInfer.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <nvjpeg.h>

#ifdef _WIN32
#ifdef EXPORT_DLL
#define DECLSPEC_API __declspec(dllexport)
#else
#define DECLSPEC_API __declspec(dllimport)
#endif
#else
#define DECLSPEC_API
#endif

static std::string cublasGetErrorString(cublasStatus_t error) {
    switch (error) {
    case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR";
#if CUDA_VERSION >= 6000
    case CUBLAS_STATUS_NOT_SUPPORTED:
        return "CUBLAS_STATUS_NOT_SUPPORTED";
#endif
#if CUDA_VERSION >= 6050
    case CUBLAS_STATUS_LICENSE_ERROR:
        return "CUBLAS_STATUS_LICENSE_ERROR";
#endif
    default:
        return "Unknown cublas status";
    }
}

static std::string NvJpegGetErrorString(nvjpegStatus_t status) {
    switch (status) {
    case NVJPEG_STATUS_NOT_INITIALIZED:
        return "NVJPEG_STATUS_NOT_INITIALIZED";
    case NVJPEG_STATUS_INVALID_PARAMETER:
        return "NVJPEG_STATUS_INVALID_PARAMETER";
    case NVJPEG_STATUS_BAD_JPEG:
        return "NVJPEG_STATUS_BAD_JPEG";
    case NVJPEG_STATUS_JPEG_NOT_SUPPORTED:
        return "NVJPEG_STATUS_JPEG_NOT_SUPPORTED";
    case NVJPEG_STATUS_ALLOCATOR_FAILURE:
        return "NVJPEG_STATUS_ALLOCATOR_FAILURE";
    case NVJPEG_STATUS_EXECUTION_FAILED:
        return "NVJPEG_STATUS_EXECUTION_FAILED";
    case NVJPEG_STATUS_ARCH_MISMATCH:
        return "NVJPEG_STATUS_ARCH_MISMATCH";
    case NVJPEG_STATUS_INTERNAL_ERROR:
        return "NVJPEG_STATUS_INTERNAL_ERROR";
    case NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED:
        return "NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED";
    default:
        return "Unknown nvjpeg error";
    }
}

#define CUDACHECK(status)                                                      \
    do {                                                                       \
        auto ret = status;                                                     \
        if (ret != 0) {                                                        \
            LOG_FATAL("CUDA ERROR: {}", cudaGetErrorString(status));           \
        }                                                                      \
    } while (0)

#define CUBLASCHECK(status)                                                    \
    do {                                                                       \
        auto ret = status;                                                     \
        if (ret != 0) {                                                        \
            LOG_FATAL("CUBLAS ERROR: {}", cublasGetErrorString(status));       \
        }                                                                      \
    } while (0)

#define CUDNNCHECK(status)                                                     \
    do {                                                                       \
        auto ret = status;                                                     \
        if (ret != 0) {                                                        \
            LOG_FATAL("CUDNN ERROR: {}", cudnnGetErrorString(status));         \
        }                                                                      \
    } while (0)

#define CUDA_DRVAPI_CALL(call)                                                 \
    do {                                                                       \
        auto errorCode = call;                                                 \
        if (errorCode != 0) {                                                  \
            const char *szErrName = nullptr;                                   \
            cuGetErrorString(errorCode, &szErrName);                           \
            LOG_FATAL("CUDA DRIVER API ERROR {}", szErrName);                  \
        }                                                                      \
    } while (0)

#define SAFE_FREE(ptr)                                                         \
    do {                                                                       \
        if (ptr) {                                                             \
            delete ptr;                                                        \
            ptr = nullptr;                                                     \
        }                                                                      \
    } while (0)

#define CUDA_FREE(ptr)                                                         \
    do {                                                                       \
        if (ptr) {                                                             \
            CUDACHECK(cudaFree(ptr));                                          \
            ptr = nullptr;                                                     \
        }                                                                      \
    } while (0)

#define CUDA_HOST_FREE(ptr)                                                    \
    do {                                                                       \
        if (ptr) {                                                             \
            CUDACHECK(cudaFreeHost(ptr));                                      \
            ptr = nullptr;                                                     \
        }                                                                      \
    } while (0)

#define NVJPEGCHECK(status)                                                    \
    do {                                                                       \
        auto ret = status;                                                     \
        if (ret != 0) {                                                        \
            LOG_FATAL("NVJPEG ERROR: {}", NvJpegGetErrorString(ret));          \
        }                                                                      \
    } while (0)