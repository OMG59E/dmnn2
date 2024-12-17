/*
 * @Author: xingwg
 * @Date: 2024-10-12 09:59:37
 * @LastEditTime: 2024-10-12 11:59:29
 * @FilePath: /dmnn2/src/plugin/common/kernels/normalizeLayer.cu
 * @Description: 
 * 
 * Copyright (c) 2024 by Chinasvt, All Rights Reserved. 
 */

#include "kernel.h"
#include "../bboxUtils.h"

size_t normalizePluginWorkspaceSize(bool acrossSpatial, int C, int H, int W) {
    if (acrossSpatial)
        return sizeof(float) * C * H * W;
    else
        return (size_t) 0;
}

template<unsigned nthds_per_cta>
__launch_bounds__(nthds_per_cta)
__global__ void normalizeNotAcrossSpatialKernel(
        const bool channelShared,
        const int N,
        const int C,
        const int H,
        const int W,
        const float eps,
        const float *scale,
        float *inputData,
        float *outputData) {
    const int dim = C * H * W;
    const int spatialDim = H * W;
    const int tile = 32;
    const int numTile = (spatialDim + tile - 1) / tile;
    for (int n = blockIdx.x; n < N * numTile; n += gridDim.x) {
        float *input = inputData + (n / numTile) * dim;
        float *output = outputData + (n / numTile) * dim;
        __shared__ float sum[tile];
        float localsum = 0.0F;
        for (int i = threadIdx.x; i < tile; i += nthds_per_cta) {
            sum[i] = 0.0F;
        }
        __syncthreads();
        for (int i = threadIdx.x; i < C * tile; i += nthds_per_cta) {
            int row = i / tile;
            int col = (n % numTile) * tile + i % tile;
            float data = 0.0F;
            if (col < spatialDim)
                data = input[row * spatialDim + col];
            localsum += data * data;
        }
        atomicAdd(&sum[threadIdx.x & 31], localsum);
        __syncthreads();
        for (int i = threadIdx.x; i < C * tile; i += nthds_per_cta) {
            int row = i / tile;
            int col = (n % numTile) * tile + i % tile;
            if (col < spatialDim) {
                int offset = row * spatialDim + col;
                output[offset] = input[offset] / sqrt(sum[threadIdx.x & 31] + eps);
            }
        }
        if (channelShared) {
            for (int i = threadIdx.x; i < C * tile; i += nthds_per_cta) {
                int row = i / tile;
                int col = (n % numTile) * tile + i % tile;
                if (col < spatialDim)
                    output[row * spatialDim + col] *= scale[0];
            }
        } else {
            for (int i = threadIdx.x; i < C * tile; i += nthds_per_cta) {
                int row = i / tile;
                int col = (n % numTile) * tile + i % tile;
                if (col < spatialDim)
                    output[row * spatialDim + col] *= scale[row];
            }
        }
    }
}

pluginStatus_t normalizeNotAcrossSpatialGpu(
        cudaStream_t stream,
        const bool channelShared,
        const int N,
        const int C,
        const int H,
        const int W,
        const float eps,
        const void *scale,
        const void *inputData,
        void *outputData) {
    const int BS = 128;
    const int GS = 256;
    // assumes warp size == 32
    if (BS % 32 != 0)
        LOG_FATAL("BS %d should be a multiple of 32", BS);
    normalizeNotAcrossSpatialKernel < BS ><<<GS, BS, 0, stream>>>(channelShared, N, C, H, W, eps, (const float *) scale, (float *) inputData, (float *) outputData);
    CUDACHECK(cudaGetLastError());
    return STATUS_SUCCESS;
}

__global__ void squareKernel(const int n, const float *x, float *y) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        y[i] = x[i] * x[i];
    }
}

__global__ void scalChannelKernel(
        const int n,
        const int spatialDim,
        const float *inputData,
        const float *scale,
        float *outputData) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        // scale factors are indepedent across different channels
        // scale[i / spatialDim]: find the right scale factor for specific channels
        outputData[i] = inputData[i] / scale[i / spatialDim];
    }
}

pluginStatus_t normalizeInference(
        cudaStream_t stream,
        cublasHandle_t handle,
        const bool acrossSpatial,
        const bool channelShared,
        const int N,
        const int C,
        const int H,
        const int W,
        const float eps,
        const void *scale,
        const void *inputData,
        void *outputData,
        void *workspace) {
    const int dim = C * H * W;
    // Normalization is conducted for each sample from the batch indepdently
    if (acrossSpatial) {
        float *input = (float *) const_cast<void *>(inputData);
        float *output = (float *) outputData;
        float *buffer = (float *) workspace;
        for (int n = 0; n < N; ++n) {
            // Take the square of each element in the input
            squareKernel << < (dim + 511) / 512, 512, 0, stream >> > (dim, input, buffer);
            float normsqr = 0.0F;
            // Sum up all the squared elements
            CUBLASCHECK(cublasSasum(handle, dim, buffer, 1, &normsqr));
            // Make a copy of the input to the output
            CUBLASCHECK(cublasScopy(handle, dim, input, 1, output, 1));
            // Calculate the inverse of the square root of the sum
            // Use eps to prevent being divided by zero
            normsqr = 1 / sqrt(normsqr + eps);
            // Scale all the outputs by normsqr
            CUBLASCHECK(cublasSscal(handle, dim, &normsqr, output, 1));
            // If channel shared is true, scale all the outputs
            if (channelShared) {
                CUBLASCHECK(cublasSscal(handle, dim, (float *) scale, output, 1));
            } else { // Use different scale factors for different channels
                // scale the output according to channels
                scalChannelKernel<<<(dim + 511) / 512, 512, 0, stream >>>(dim, H * W, output, (float *) scale, output);
            }
            // Move cursors
            input += dim;
            output += dim;
        }
        return STATUS_SUCCESS;
    } else { // Normalization ignoring the batch
        return normalizeNotAcrossSpatialGpu(stream, channelShared, N, C, H, W, eps, scale, inputData, outputData);
    }
}
