/*
 * @Author: xingwg
 * @Date: 2024-10-12 09:59:37
 * @LastEditTime: 2024-10-12 11:40:23
 * @FilePath: /dmnn2/src/plugin/scaleV2Plugin/scaleV2Plugin.cu
 * @Description: 
 * 
 * Copyright (c) 2024 by Chinasvt, All Rights Reserved. 
 */
#include "scaleV2Plugin.h"
#include "base_types.h"

using namespace nvinfer1::plugin;

template <typename Dtype>
__global__ void ScaleForward(const int n, const Dtype* in,
                             const Dtype* scale, const int scale_dim, const int inner_dim, Dtype* out) {
    CUDA_KERNEL_LOOP(idx, n) {
        const int scale_index = (idx / inner_dim) % scale_dim;
        out[idx] = in[idx] * scale[scale_index];
    }
}

int ScaleV2::enqueue(int32_t batchSize, void const* const* inputs, void* const* outputs,
        void* workspace, cudaStream_t stream) noexcept {

    int count = batchSize * channel_in_ * height_in_ * width_in_;
    int scale_dim = batchSize * channel_in_;
    int inner_dim = height_in_ * width_in_;
    if (data_type_ == DataType::kFLOAT) {
        auto top_data = reinterpret_cast<float*>(outputs[0]);
        const auto bottom_data = reinterpret_cast<const float *>(inputs[0]);
        const auto scale_data = reinterpret_cast<const float *>(inputs[1]);
        ScaleForward << < CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream >> > (count, bottom_data, scale_data, scale_dim, inner_dim, top_data);
    } else if (data_type_ == DataType::kHALF) {
        auto top_data = reinterpret_cast<__half *>(outputs[0]);
        const auto bottom_data = reinterpret_cast<const __half *>(inputs[0]);
        const auto scale_data = reinterpret_cast<const __half *>(inputs[1]);
        ScaleForward<< < CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream >> > (count, bottom_data, scale_data, scale_dim, inner_dim, top_data);
    } else if (data_type_ == DataType::kINT8) {
        LOG_FATAL("Not support int8 dataType");
    };

    return 0;
}