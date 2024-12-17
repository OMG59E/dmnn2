/*
 * @Author: xingwg
 * @Date: 2024-10-12 09:59:37
 * @LastEditTime: 2024-10-12 10:14:42
 * @FilePath: /dmnn2/src/plugin/common/kernels/common.cu
 * @Description: 
 * 
 * Copyright (c) 2024 by Chinasvt, All Rights Reserved. 
 */
#include <cublas_v2.h>
#include <cub/cub.cuh>
#include <stdint.h>
#include "kernel.h"
#include "../bboxUtils.h"

#define CUDA_MEM_ALIGN 256

// HASH
//unsigned int hash(const void* array_, size_t size)
//{
//    // Apply hashing only when debugging RPN codes.
//    if (DEBUG_ENABLE)
//    {
//        const char* array_const;
//        char* array;
//        cudaMallocHost((void**) &array, size);
//        cudaMemcpy(array, array_, size, cudaMemcpyDeviceToHost);
//        array_const = array;
//        unsigned int hash = 45599;
//        for (size_t i = 0; i < size; i++)
//        {
//            unsigned int value = array_const[i];
//            hash = hash * 1487 + value;
//            hash = hash * 317;
//            hash = hash % 105359;
//        }
//        return hash;
//    }
//    else
//    {
//        return 0;
//    }
//}

// ALIGNPTR
int8_t *alignPtr(int8_t *ptr, uintptr_t to) {
    uintptr_t addr = (uintptr_t) ptr;
    if (addr % to) {
        addr += to - addr % to;
    }
    return (int8_t *) addr;
}

// NEXTWORKSPACEPTR
int8_t *nextWorkspacePtr(int8_t *ptr, uintptr_t previousWorkspaceSize) {
    uintptr_t addr = (uintptr_t) ptr;
    addr += previousWorkspaceSize;
    return alignPtr((int8_t *) addr, CUDA_MEM_ALIGN);
}

// CALCULATE TOTAL WORKSPACE SIZE
size_t calculateTotalWorkspaceSize(size_t * workspaces, int count) {
    size_t total = 0;
    for (int i = 0; i<count; i++) {
        total += workspaces[i];
        if (workspaces[i] % CUDA_MEM_ALIGN) {
            total += CUDA_MEM_ALIGN - (workspaces[i] % CUDA_MEM_ALIGN);
        }
    }
    return total;
}

using nvinfer1::DataType;

// DATA TYPE SIZE
size_t dataTypeSize(const DataType dtype) {
    switch (dtype) {
        case DataType::kINT8:
            return sizeof(char);
        case DataType::kHALF:
            return sizeof(short);
        case DataType::kFLOAT:
            return sizeof(float);
        default:
            return 0;
    }
}


template<unsigned nthds_per_cta>
__launch_bounds__(nthds_per_cta)
__global__ void setUniformOffsets_kernel(const int num_segments, const int offset, int *d_offsets) {
    const int idx = blockIdx.x * nthds_per_cta + threadIdx.x;
    if (idx <= num_segments)
        d_offsets[idx] = idx * offset;
}

void setUniformOffsets(cudaStream_t stream, const int num_segments, const int offset, int *d_offsets) {
    const int BS = 32;
    const int GS = (num_segments + 1 + BS - 1) / BS;
    setUniformOffsets_kernel < BS ><<<GS, BS, 0, stream>>>(num_segments, offset, d_offsets);
}