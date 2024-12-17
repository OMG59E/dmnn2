/*
 * @Author: xingwg
 * @Date: 2024-10-21 16:13:35
 * @LastEditTime: 2024-10-22 19:14:18
 * @FilePath: /dmnn2/src/codecs/jpeg_dec.cc
 * @Description:
 *
 * Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#include "codecs/jpeg_dec.h"

namespace nv {
JpegDecoder::JpegDecoder() {
    CUDACHECK(cudaStreamCreate(&stream_));
    NVJPEGCHECK(nvjpegCreateSimple(&handle_));
    NVJPEGCHECK(nvjpegJpegStateCreate(handle_, &jpeg_state_));
}

JpegDecoder::~JpegDecoder() {
    NVJPEGCHECK(nvjpegJpegStateDestroy(jpeg_state_));
    NVJPEGCHECK(nvjpegDestroy(handle_));
    CUDACHECK(cudaStreamDestroy(stream_));
}

int JpegDecoder::Decode(const std::string &filename, nv::Image &image) {
    std::ifstream input(filename,
                        std::ios::in | std::ios::binary | std::ios::ate);
    if (!input.is_open()) {
        LOG_ERROR("Failed to open image");
        return -1;
    }
    std::streamsize file_size = input.tellg();
    input.seekg(0, std::ios::beg);
    std::vector<char> data;
    data.resize(static_cast<size_t>(file_size));
    input.read(data.data(), file_size);
    return Decode(data, image);
}

int JpegDecoder::Decode(const std::vector<char> &data, nv::Image &image) {
    if (image.gpu_data) {
        LOG_ERROR("image gpu_data must be nullptr");
        return -1;
    }
    int widths[NVJPEG_MAX_COMPONENT];
    int heights[NVJPEG_MAX_COMPONENT];
    int channels;
    nvjpegChromaSubsampling_t subsampling;
    nvjpegStatus_t status = nvjpegGetImageInfo(
        handle_, reinterpret_cast<const uint8_t *>(data.data()), data.size(),
        &channels, &subsampling, widths, heights);
    if (status != NVJPEG_STATUS_SUCCESS) {
        LOG_ERROR("NVJPEG ERROR: {}", NvJpegGetErrorString(status));
        return -2;
    }
    LOG_ASSERT(channels == 3 || channels == 1);
    image.create(heights[0], widths[0], true,
                 channels != 1 ? COLOR_TYPE_BGR888_PLANAR : COLOR_TYPE_GRAY,
                 DATA_TYPE_UINT8);
    LOG_INFO("Jpeg decode alloc gpu memory: {} Bytes", image.size_bytes());
    nvjpegImage_t nvjpegImage;
    nvjpegImage.channel[0] =
        (unsigned char *)(image.gpu_data) + 0 * image.h() * image.w();
    nvjpegImage.channel[1] =
        (unsigned char *)(image.gpu_data) + 1 * image.h() * image.w();
    nvjpegImage.channel[2] =
        (unsigned char *)(image.gpu_data) + 2 * image.h() * image.w();
    nvjpegImage.pitch[0] = static_cast<unsigned int>(image.w());
    nvjpegImage.pitch[1] = static_cast<unsigned int>(image.w());
    nvjpegImage.pitch[2] = static_cast<unsigned int>(image.w());
    nvjpegOutputFormat_t nvjpegOutputFormat = NVJPEG_OUTPUT_BGR;
    status = nvjpegDecode(
        handle_, jpeg_state_, reinterpret_cast<const uint8_t *>(data.data()),
        data.size(), nvjpegOutputFormat, &nvjpegImage, stream_);
    if (status != NVJPEG_STATUS_SUCCESS) {
        LOG_ERROR("NVJPEG ERROR: {}", NvJpegGetErrorString(status));
        image.free();
        return -4;
    }
    CUDACHECK(cudaStreamSynchronize(stream_));
    CUDACHECK(cudaGetLastError());
    return 0;
}
}  // namespace nv