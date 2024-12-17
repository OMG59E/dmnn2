/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <algorithm>
#include <stdexcept>

#include <cuda_fp16.h>

#include "instanceNormalizationPlugin.h"

using namespace nvinfer1::plugin;

template<typename T, int32_t THREADS_PER_CTA>
__global__ __launch_bounds__(THREADS_PER_CTA) void in3dReluActivation(T *__restrict dst, T *__restrict src, float alpha, int32_t count) {
    int32_t idx = blockIdx.x * THREADS_PER_CTA + threadIdx.x;
    if (idx >= count)
        return;
    float val = src[idx];
    dst[idx] = (val < 0.f) ? val * alpha : val;
}

cudnnStatus_t convertTrt2cudnnDtype(nvinfer1::DataType trt_dtype, cudnnDataType_t *cudnn_dtype) {
    switch (trt_dtype) {
        case nvinfer1::DataType::kFLOAT:
            *cudnn_dtype = CUDNN_DATA_FLOAT;
            break;
        case nvinfer1::DataType::kHALF:
            *cudnn_dtype = CUDNN_DATA_HALF;
            break;
        default:
            return CUDNN_STATUS_BAD_PARAM;
    }
    return CUDNN_STATUS_SUCCESS;
}

int32_t InstanceNormalizationPlugin::enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
                                             const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs,
                                             void *const *outputs, void *workspace,
                                             cudaStream_t stream) noexcept {
    nvinfer1::Dims input_dims = inputDesc[0].dims;
    // early return for empty tensor
    if (std::any_of(input_dims.d, input_dims.d + input_dims.nbDims, [](int32_t d) { return d == 0; })) {
        return 0;
    }

    const auto callRelu = [this, &stream](void *inOut, int32_t count, nvinfer1::DataType type) {
        if (mRelu > 0) {
            const int32_t kBLOCK_SZ = 256;
            switch (type) {
                case nvinfer1::DataType::kFLOAT:
                    in3dReluActivation < float, kBLOCK_SZ ><<<(count + kBLOCK_SZ - 1) / kBLOCK_SZ, kBLOCK_SZ, 0, stream>>>(static_cast<float *>(inOut), static_cast<float *>(inOut), mAlpha, count);
                    break;
                case nvinfer1::DataType::kHALF:
                    in3dReluActivation < __half, kBLOCK_SZ ><<<(count + kBLOCK_SZ - 1) / kBLOCK_SZ, kBLOCK_SZ, 0, stream>>>(static_cast<__half *>(inOut), static_cast<__half *>(inOut), mAlpha, count);
                    break;
                default:
                    assert(0);
            }
        }
    };

    if (input_dims.nbDims <= 4) {
        nvinfer1::Dims input_dims = inputDesc[0].dims;
        int32_t n = input_dims.d[0];
        int32_t c = input_dims.d[1];
        int32_t h = input_dims.d[2];
        int32_t w = input_dims.nbDims > 3 ? input_dims.d[3] : 1;
        size_t nchan_bytes = c * sizeof(float);

        float *_d_array = static_cast<float *>(workspace);
        float *d_scale = &_d_array[0];
        float *d_bias = &_d_array[n * c];
        for (int32_t i = 0; i < n; ++i) {
            CUDACHECK(cudaMemcpyAsync(d_scale + i * c, mDeviceScale, nchan_bytes, cudaMemcpyDeviceToDevice, stream));
            CUDACHECK(cudaMemcpyAsync(d_bias + i * c, mDeviceBias, nchan_bytes, cudaMemcpyDeviceToDevice, stream));
        }

        CUDNNCHECK(cudnnSetTensor4dDescriptor(mBDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, n * c, 1, 1));
        cudnnDataType_t cudnn_dtype{};
        CUDNNCHECK(convertTrt2cudnnDtype(inputDesc[0].type, &cudnn_dtype));
        CUDNNCHECK(cudnnSetTensor4dDescriptor(mXDescriptor, CUDNN_TENSOR_NCHW, cudnn_dtype, 1, n * c, h, w));
        CUDNNCHECK(cudnnSetTensor4dDescriptor(mYDescriptor, CUDNN_TENSOR_NCHW, cudnn_dtype, 1, n * c, h, w));
        float alpha = 1;
        float beta = 0;
        void const *x_ptr = inputs[0];
        void *y_ptr = outputs[0];
        CUDNNCHECK(cudnnSetStream(mCudnnHandle, stream));
        // Note: Use of CUDNN_BATCHNORM_SPATIAL_PERSISTENT can cause numerical
        //       overflows (NaNs) for fp32 data in some circumstances. The lower-
        //       performance CUDNN_BATCHNORM_SPATIAL should be used if this is not
        //       acceptable.
        CUDNNCHECK(cudnnBatchNormalizationForwardTraining(mCudnnHandle, CUDNN_BATCHNORM_SPATIAL_PERSISTENT, &alpha,
                                                           &beta, mXDescriptor, x_ptr, mYDescriptor, y_ptr,
                                                           mBDescriptor, d_scale, d_bias, 1., nullptr, nullptr,
                                                           mEpsilon, nullptr, nullptr));

        callRelu(y_ptr, n * c * h * w, inputDesc[0].type);
    } else {
        if (inputDesc[0].format == nvinfer1::PluginFormat::kLINEAR) {
            CUDNNCHECK(cudnnSetStream(mCudnnHandle, stream));
            nvinfer1::Dims input_dims = inputDesc[0].dims;
            int32_t n = input_dims.d[0];
            int32_t c = input_dims.d[1];
            int32_t d = input_dims.d[2];
            int32_t h = input_dims.d[3];
            int32_t w = input_dims.d[4];
            size_t nchan_bytes = c * sizeof(float);

            // Note: We repeat the data for each batch entry so that we can do the full
            //       computation in a single CUDNN call in enqueue().
            float *_d_array = (float *) workspace;
            float *d_scale = &_d_array[0];
            float *d_bias = &_d_array[n * c];
            for (int32_t i = 0; i < n; ++i) {
                CUDACHECK(cudaMemcpyAsync(d_scale + i * c, mDeviceScale, nchan_bytes, cudaMemcpyDeviceToDevice, stream));
                CUDACHECK(cudaMemcpyAsync(d_bias + i * c, mDeviceBias, nchan_bytes, cudaMemcpyDeviceToDevice, stream));
            }

            int32_t nc_dimA[] = {1, n * c, 1, 1, 1};
            int32_t nc_strideA[] = {nc_dimA[1] * nc_dimA[2] * nc_dimA[3] * nc_dimA[4],
                                    nc_dimA[2] * nc_dimA[3] * nc_dimA[4], nc_dimA[3] * nc_dimA[4], nc_dimA[4], 1};
            int32_t img_dimA[] = {1, n * c, d, h, w};
            int32_t img_strideA[] = {img_dimA[1] * img_dimA[2] * img_dimA[3] * img_dimA[4],
                                     img_dimA[2] * img_dimA[3] * img_dimA[4], img_dimA[3] * img_dimA[4], img_dimA[4],
                                     1};

            CUDNNCHECK(cudnnSetTensorNdDescriptor(mBDescriptor, CUDNN_DATA_FLOAT, 5, nc_dimA, nc_strideA));
            cudnnDataType_t cudnn_dtype;
            CUDNNCHECK(convertTrt2cudnnDtype(inputDesc[0].type, &cudnn_dtype));
            CUDNNCHECK(cudnnSetTensorNdDescriptor(mXDescriptor, cudnn_dtype, 5, img_dimA, img_strideA));
            CUDNNCHECK(cudnnSetTensorNdDescriptor(mYDescriptor, cudnn_dtype, 5, img_dimA, img_strideA));
            float alpha = 1;
            float beta = 0;

            void const *x_ptr = inputs[0];
            void *y_ptr = outputs[0];
            // Note: Use of CUDNN_BATCHNORM_SPATIAL_PERSISTENT can cause numerical
            //       overflows (NaNs) for fp32 data in some circumstances. The lower-
            //       performance CUDNN_BATCHNORM_SPATIAL should be used if this is not
            //       acceptable.
            CUDNNCHECK(cudnnBatchNormalizationForwardTraining(mCudnnHandle, CUDNN_BATCHNORM_SPATIAL_PERSISTENT, &alpha,
                                                               &beta, mXDescriptor, x_ptr, mYDescriptor, y_ptr,
                                                               mBDescriptor, d_scale, d_bias, 1., nullptr, nullptr,
                                                               mEpsilon, nullptr, nullptr));

            callRelu(y_ptr, n * c * d * h * w, inputDesc[0].type);
        } else if (inputDesc[0].format == nvinfer1::PluginFormat::kDHWC8
                   || inputDesc[0].format == nvinfer1::PluginFormat::kCDHW32) {
            int32_t input_data_type = (inputDesc[0].type == nvinfer1::DataType::kHALF) ? 1 : 2;
            int32_t output_data_type = (outputDesc[0].type == nvinfer1::DataType::kHALF) ? 1 : 2;

            nvinfer1::Dims input_dims = inputDesc[0].dims;
            int32_t n = input_dims.d[0];
            int32_t c = input_dims.d[1];
            int32_t d = input_dims.d[2];
            int32_t h = input_dims.d[3];
            int32_t w = input_dims.d[4];

            InstanceNormFwdParams params;
            params.nhw = d * h * w;
            params.c = c;
            params.n = n;

            size_t size_sums, size_counts, size_retired_ctas;
            instanceNormBufferSizesDispatch(
                    mContext, params, size_sums, size_counts, size_retired_ctas, input_data_type, output_data_type);

            size_t size_nc = n * c * sizeof(float);
            size_nc = ((size_nc + 256 - 1) / 256) * 256;

            char *d_buf = static_cast<char *>(workspace);

            params.gmem_sums = reinterpret_cast<GMEM_SUMS_TYPE *>(d_buf);
            d_buf += size_sums;
            params.gmem_counts = reinterpret_cast<int32_t *>(d_buf);
            d_buf += size_counts;
            params.gmem_retired_ctas = reinterpret_cast<int32_t *>(d_buf);
            d_buf += size_retired_ctas;
            params.gmem_running_mean = reinterpret_cast<float *>(d_buf);
            d_buf += size_nc;
            params.gmem_running_var = reinterpret_cast<float *>(d_buf);
            d_buf += size_nc;
            params.gmem_saved_mean = reinterpret_cast<float *>(d_buf);
            d_buf += size_nc;
            params.gmem_saved_var = reinterpret_cast<float *>(d_buf);
            d_buf += size_nc;

            params.gmem_src = inputs[0];
            params.gmem_dst = outputs[0];
            params.gmem_bias = mDeviceBias;
            params.gmem_scale = mDeviceScale;

            params.var_eps = mEpsilon;
            params.exp_avg_factor = 1.F; //(float)exp_avg_factor;
            params.use_relu = mRelu;     // use_relu;
            params.relu_alpha = mAlpha;  // relu_alpha;

            params.in_scale = inputDesc[0].scale;
            assert(outputDesc[0].scale != 0.F);
            params.out_scale = 1.F / outputDesc[0].scale;

            instanceNormFwdDispatch(mContext, params, stream, input_data_type, output_data_type);
        } else {
            assert(false && "Unexpected input format");
        }
    }
    return 0;
}
