
#include "focusPlugin.h"

using namespace nvinfer1::plugin;

template <typename T>
__global__ void focus_kernel(const int nbThreads,
                             const T* input, const int iChannels, const int iHeight, const int iWidth,
                             T* output, const int oChannels, const int oHeight, const int oWidth) {
    CUDA_KERNEL_LOOP(idx, nbThreads) {
        // n*c*h*w
        int dn = idx / oWidth / oHeight / oChannels;
        int oc = idx / oWidth / oHeight % oChannels; //0-11
        int oh = idx / oWidth % oHeight; //0-319
        int ow = idx % oWidth; //0-319
        int size = iChannels * iHeight * iWidth;

        output[((dn * size) + (0 * oHeight * oWidth)) + (oc % 3 * oHeight * oWidth) + (oh * oWidth) + ow] = input[(dn * size) + (oc % 3 * iHeight * iWidth) + (2 * oh + 0) * iWidth + (2 * ow + 0)];
        output[((dn * size) + (3 * oHeight * oWidth)) + (oc % 3 * oHeight * oWidth) + (oh * oWidth) + ow] = input[(dn * size) + (oc % 3 * iHeight * iWidth) + (2 * oh + 1) * iWidth + (2 * ow + 0)];
        output[((dn * size) + (6 * oHeight * oWidth)) + (oc % 3 * oHeight * oWidth) + (oh * oWidth) + ow] = input[(dn * size) + (oc % 3 * iHeight * iWidth) + (2 * oh + 0) * iWidth + (2 * ow + 1)];
        output[((dn * size) + (9 * oHeight * oWidth)) + (oc % 3 * oHeight * oWidth) + (oh * oWidth) + ow] = input[(dn * size) + (oc % 3 * iHeight * iWidth) + (2 * oh + 1) * iWidth + (2 * ow + 1)];
    }
}

int Focus::enqueue(int32_t batchSize, void const* const* inputs, void* const* outputs, void *workspace,
                   cudaStream_t stream) noexcept {
    const int iChannels = input_shape_.c();
    const int iHeight = input_shape_.h();
    const int iWidth = input_shape_.w();
    const int oChannels = iChannels * 4;
    const int oHeight = iHeight / 2;
    const int oWidth = iWidth / 2;
    const int nbThreads = batchSize*iChannels*iHeight*iWidth;
    if (data_type_ == DataType::kFLOAT) {
        auto* bottom_data = reinterpret_cast<const float*>(inputs[0]);
        auto* top_data = reinterpret_cast<float*>(outputs[0]);
        focus_kernel<<<CUDA_GET_BLOCKS(nbThreads), CUDA_NUM_THREADS, 0, stream>>>(nbThreads, bottom_data, iChannels, iHeight, iWidth, top_data, oChannels, oHeight, oWidth);
    } else if (data_type_ == DataType::kHALF) {
        auto* bottom_data = reinterpret_cast<const __half*>(inputs[0]);
        auto* top_data = reinterpret_cast<__half*>(outputs[0]);
        focus_kernel<<<CUDA_GET_BLOCKS(nbThreads), CUDA_NUM_THREADS, 0, stream>>>(nbThreads, bottom_data, iChannels, iHeight, iWidth, top_data, oChannels, oHeight, oWidth);
    } else if (data_type_ == DataType::kINT8) {
        auto* bottom_data = reinterpret_cast<const int8_t*>(inputs[0]);
        auto* top_data = reinterpret_cast<int8_t*>(outputs[0]);
        focus_kernel<<<CUDA_GET_BLOCKS(nbThreads), CUDA_NUM_THREADS, 0, stream>>>(nbThreads, bottom_data, iChannels, iHeight, iWidth, top_data, oChannels, oHeight, oWidth);
    };
    CUDACHECK(cudaDeviceSynchronize());
    CUDACHECK(cudaGetLastError());
    return 0;
}