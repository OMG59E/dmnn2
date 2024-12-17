/*
 * @Author: xingwg
 * @Date: 2024-10-12 09:59:37
 * @LastEditTime: 2024-10-12 14:28:20
 * @FilePath: /dmnn2/src/plugin/YOLOBoxPlugin/YOLOBox.cu
 * @Description: 
 * 
 * Copyright (c) 2024 by Chinasvt, All Rights Reserved. 
 */

#include "YOLOBoxPlugin.h"

using namespace nvinfer1::plugin;

__global__ void yolo_box_kernel(
        const int nbThreads,
        const float *anchors,
        const int num_anchors,
        const int stride,
        const int N,
        const int C,
        const int H,
        const int W,
        const int layer_h,
        const float *inputData,
        float *outputData) {
    // batchsize * 3 * 6400/1600/400 * 9
    CUDA_KERNEL_LOOP(idx, nbThreads) {
        int dc = (idx / H) % C;  // 0 - N*C
        int dh = idx % H;  // 0 - H

        int dx = dh % layer_h;
        int dy = dh / layer_h;

        outputData[idx * W + 0] = (inputData[idx * W + 0] * 2.0f - 0.5f + dx) * stride;
        outputData[idx * W + 1] = (inputData[idx * W + 1] * 2.0f - 0.5f + dy) * stride;
        outputData[idx * W + 2] = inputData[idx * W + 2] * inputData[idx * W + 2] * 4.0f * anchors[2 * dc + 0];
        outputData[idx * W + 3] = inputData[idx * W + 3] * inputData[idx * W + 3] * 4.0f * anchors[2 * dc + 1];
        outputData[idx * W + 4] = inputData[idx * W + 4];
        outputData[idx * W + 5] = inputData[idx * W + 5];
        outputData[idx * W + 6] = inputData[idx * W + 6];
        outputData[idx * W + 7] = inputData[idx * W + 7];
        outputData[idx * W + 8] = inputData[idx * W + 8];
    }
}

pluginStatus_t YOLOBoxInference(cudaStream_t stream,
                                const void *anchors,
                                const int num_anchors,
                                const int stride,
                                const int N,
                                const int C,
                                const int H,
                                const int W,
                                const void *inputData,
                                void *outputData) {
    int layer_h = int(sqrt(H));
    const int nbThreads = N * C * H; // 3 * 6400 3 * 1600 3 * 400
    yolo_box_kernel << < CUDA_GET_BLOCKS(nbThreads), CUDA_NUM_THREADS >> >
        (nbThreads, (const float*)anchors, num_anchors, stride, N, C, H, W, layer_h, (const float*)inputData, (float*)outputData);
    CUDACHECK(cudaGetLastError());
    return STATUS_SUCCESS;
}

int YOLOBox::enqueue(int32_t batchSize, void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept {
    void *outputData = outputs[0];
    const auto *inputData = reinterpret_cast<const float *>(inputs[0]);
    assert(mAnchors_.count == 6);
    pluginStatus_t status = YOLOBoxInference(stream, mAnchors_.values, mAnchors_.count, mParam_.stride, batchSize, C_, H_, W_, inputData, outputData);
    assert(status == STATUS_SUCCESS);
    return 0;
}