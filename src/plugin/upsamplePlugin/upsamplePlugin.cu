/*
 * @Author: xingwg
 * @Date: 2024-10-12 09:59:37
 * @LastEditTime: 2024-10-12 12:09:51
 * @FilePath: /dmnn2/src/plugin/upsamplePlugin/upsamplePlugin.cu
 * @Description: 
 * 
 * Copyright (c) 2024 by Chinasvt, All Rights Reserved. 
 */
#include "upsamplePlugin.h"
#include "base_types.h"

using namespace nvinfer1::plugin;

__device__ int translate_idx(int ii, int d1, int d2, int d3, int scale_factor) {
    int x, y, z, w;
    w = ii % d3;
    ii = ii / d3;
    z = ii % d2;
    ii = ii / d2;
    y = ii % d1;
    ii = ii / d1;
    x = ii;
    w = w / scale_factor;
    z = z / scale_factor;
    d2 /= scale_factor;
    d3 /= scale_factor;
    return (((x * d1 + y) * d2) + z) * d3 + w;
}

template <typename Dtype>
__global__ void upscale(const Dtype *input, Dtype *output, int no_elements, int scale_factor, int d1, int d2, int d3) {
    int ii = threadIdx.x + blockDim.x * blockIdx.x;
    if (ii >= no_elements)
        return;
    int ipidx = translate_idx(ii, d1, d2, d3, scale_factor);
    output[ii] = input[ipidx];
}

void UpsampleForward(cudaStream_t stream, const float* bottom, const int count, const int scale,
        const int oC, const int oH, const int oW, float* top) {
    upscale<float><<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(bottom, top, count, scale, oC, oH, oW);
    CUDACHECK(cudaStreamSynchronize(stream));
}

int Upsample::enqueue(int batchSize, void const* const* inputs,
        void* const* outputs, void *workspace, cudaStream_t stream) noexcept {
    const auto* bottom = static_cast<const float*>(inputs[0]);
    auto* top = static_cast<float*>(outputs[0]);
    const int oC = mOutputDims.d[0];
    const int oH = mOutputDims.d[1];
    const int oW = mOutputDims.d[2];
    const int count = batchSize*oC*oH*oW;
    UpsampleForward(stream, bottom, count, mScale, oC, oH, oW, top);
    return 0;
};