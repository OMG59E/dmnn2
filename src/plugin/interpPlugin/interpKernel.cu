/*
 * @Author: xingwg
 * @Date: 2024-10-12 09:59:37
 * @LastEditTime: 2024-10-12 11:36:58
 * @FilePath: /dmnn2/src/plugin/interpPlugin/interpKernel.cu
 * @Description: 
 * 
 * Copyright (c) 2024 by Chinasvt, All Rights Reserved. 
 */

#include "interpPlugin.h"
#include "error_check.h"
#include "base_types.h"

using namespace nvinfer1::plugin;

// Bi-linear interpolation
// IN : [channels height1 width1] cropped from a bigger [Height1 Width1] image
// OUT: [channels height2 width2] cropped from a bigger [Height2 Width2] image
template <typename Dtype, bool packed>
__global__ void caffe_gpu_interp2_kernel(const int n, const float rheight, const float rwidth,
                                         const int channels,
                                         const Dtype *data1, const int x1, const int y1, const int height1, const int width1, const int Height1, const int Width1,
                                         Dtype *data2, const int x2, const int y2, const int height2, const int width2, const int Height2, const int Width2) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        const int w2 = index % width2; // 0:width2-1
        const int h2 = index / width2; // 0:height2-1
        // special case: just copy
        if (height1 == height2 && width1 == width2) {
            const int h1 = h2;
            const int w1 = w2;
            if (packed) {
                const Dtype *pos1 = &data1[channels * ((y1 + h1) * Width1 + (x1 + w1))];
                Dtype *pos2 = &data2[channels * ((y2 + h2) * Width2 + (x2 + w2))];
                for (int c = 0; c < channels; ++c)
                {
                    pos2[0] = pos1[0];
                    pos1++;
                    pos2++;
                }
            } else {
                const Dtype *pos1 = &data1[(y1 + h1) * Width1 + (x1 + w1)];
                Dtype *pos2 = &data2[(y2 + h2) * Width2 + (x2 + w2)];
                for (int c = 0; c < channels; ++c) {
                    pos2[0] = pos1[0];
                    pos1 += Width1 * Height1;
                    pos2 += Width2 * Height2;
                }
            }
            return;
        }
        //
        const float h1r = rheight * h2;
        const int h1 = h1r;
        const int h1p = (h1 < height1 - 1) ? 1 : 0;
        const Dtype h1lambda = h1r - h1;
        const Dtype h0lambda = Dtype(1.) - h1lambda;
        //
        const float w1r = rwidth * w2;
        const int w1 = w1r;
        const int w1p = (w1 < width1 - 1) ? 1 : 0;
        const Dtype w1lambda = w1r - w1;
        const Dtype w0lambda = Dtype(1.) - w1lambda;
        //
        if (packed) {
            const Dtype *pos1 = &data1[channels * ((y1 + h1) * Width1 + (x1 + w1))];
            Dtype *pos2 = &data2[channels * ((y2 + h2) * Width2 + (x2 + w2))];
            for (int c = 0; c < channels; ++c)
            {
                pos2[0] =
                        h0lambda * (w0lambda * pos1[0] + w1lambda * pos1[channels * w1p]) +
                        h1lambda *
                        (w0lambda * pos1[channels * h1p * Width1] + w1lambda * pos1[channels * (h1p * Width1 + w1p)]);
                pos1++;
                pos2++;
            }
        } else {
            const Dtype *pos1 = &data1[(y1 + h1) * Width1 + (x1 + w1)];
            Dtype *pos2 = &data2[(y2 + h2) * Width2 + (x2 + w2)];
            for (int c = 0; c < channels; ++c)
            {
                pos2[0] =
                        h0lambda * (w0lambda * pos1[0] + w1lambda * pos1[w1p]) +
                        h1lambda * (w0lambda * pos1[h1p * Width1] + w1lambda * pos1[h1p * Width1 + w1p]);
                pos1 += Width1 * Height1;
                pos2 += Width2 * Height2;
            }
        }
    }
}


void caffe_gpu_interp2(cudaStream_t stream,
                       const int channels,
                       const float *data1, const int x1, const int y1, const int height1, const int width1, const int Height1, const int Width1,
                       float *data2, const int x2, const int y2, const int height2, const int width2, const int Height2, const int Width2) {
    assert(x1 >= 0 && y1 >= 0 && height1 > 0 && width1 > 0 && x2 >= 0 && y2 >= 0 && height2 > 0 && width2 > 0);
    assert(Width1 >= width1 + x1 && Height1 >= height1 + y1 && Width2 >= width2 + x2 && Height2 >= height2 + y2);
    const float rheight = (height2 > 1) ? static_cast<float>(height1 - 1) / (height2 - 1) : 0.f;
    const float rwidth = (width2 > 1) ? static_cast<float>(width1 - 1) / (width2 - 1) : 0.f;
    const int num_kernels = height2 * width2;
    caffe_gpu_interp2_kernel<float, false><<<CUDA_GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>
                                                                            (num_kernels, rheight, rwidth, channels,
                                                                                    data1, x1, y1, height1, width1, Height1, Width1,
                                                                                    data2, x2, y2, height2, width2, Height2, Width2);
    CUDACHECK(cudaDeviceSynchronize());
    CUDACHECK(cudaPeekAtLastError());
}

int Interp::enqueue(int32_t batchSize, const void* const* inputs,
        void* const* outputs, void*, cudaStream_t stream) noexcept {
    auto* output = reinterpret_cast<float*>(outputs[0]);
    const auto* input = reinterpret_cast<const float*>(inputs[0]);

    auto height_in_eff = mIHeight + mPadBegin + mPadEnd;
    auto width_in_eff = mIWidth + mPadBegin + mPadEnd;

    caffe_gpu_interp2(stream, batchSize * mChannels, input, -mPadBegin, -mPadBegin, height_in_eff, width_in_eff, mIHeight, mIWidth,
                      output, 0, 0, mOHeight, mOWidth, mOHeight, mOWidth);
    return 0;
}


