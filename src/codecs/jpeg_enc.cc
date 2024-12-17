/*
 * @Author: xingwg
 * @Date: 2024-10-21 17:44:47
 * @LastEditTime: 2024-10-23 09:15:16
 * @FilePath: /dmnn2/src/codecs/jpeg_enc.cc
 * @Description:
 *
 * Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#include "codecs/jpeg_enc.h"
#include "utils/to_bin.h"

namespace nv {
JpegEncoder::JpegEncoder() {
    CUDACHECK(cudaStreamCreate(&stream_));
    NVJPEGCHECK(nvjpegCreateSimple(&handle_));
    NVJPEGCHECK(nvjpegEncoderStateCreate(handle_, &enc_state_, stream_));
    NVJPEGCHECK(nvjpegEncoderParamsCreate(handle_, &enc_params_, stream_));
    NVJPEGCHECK(nvjpegEncoderParamsSetQuality(enc_params_, 70, stream_));
    NVJPEGCHECK(
        nvjpegEncoderParamsSetOptimizedHuffman(enc_params_, 0, stream_));
    NVJPEGCHECK(nvjpegEncoderParamsSetSamplingFactors(enc_params_,
                                                      NVJPEG_CSS_420, stream_));
}

JpegEncoder::~JpegEncoder() {
    NVJPEGCHECK(nvjpegEncoderParamsDestroy(enc_params_));
    NVJPEGCHECK(nvjpegEncoderStateDestroy(enc_state_));
    NVJPEGCHECK(nvjpegDestroy(handle_));
    CUDACHECK(cudaStreamDestroy(stream_));
}

void JpegEncoder::SetQuality(int quality) {
    NVJPEGCHECK(nvjpegEncoderParamsSetQuality(enc_params_, quality, stream_));
}

void JpegEncoder::SetOptimizedHuffman(bool bUseHuffman) {
    NVJPEGCHECK(nvjpegEncoderParamsSetOptimizedHuffman(enc_params_, bUseHuffman,
                                                       stream_));
}

void JpegEncoder::SetSamplingFactors(nvjpegChromaSubsampling_t subsampling) {
    NVJPEGCHECK(nvjpegEncoderParamsSetSamplingFactors(enc_params_, subsampling,
                                                      stream_));
}

int JpegEncoder::Encode(const std::string &savePath, const nv::Image &img) {
    auto data = Encode(img);
    if (data.empty()) {
        LOG_ERROR("encode image data is empty");
        return -1;
    }
    to_bin(data.data(), data.size(), savePath.c_str());
    return 0;
}

std::vector<char> JpegEncoder::Encode(const nv::Image &img) {
    LOG_ASSERT(img.colorType == COLOR_TYPE_BGR888_PLANAR ||
               img.colorType == COLOR_TYPE_RGB888_PLANAR);
    nvjpegImage_t nvjpegImage;
    nvjpegImage.pitch[0] = static_cast<unsigned int>(img.w());
    nvjpegImage.pitch[1] = static_cast<unsigned int>(img.w());
    nvjpegImage.pitch[2] = static_cast<unsigned int>(img.w());
    nvjpegImage.channel[0] = (uint8_t *)(img.gpu_data) + 0 * img.h() * img.w();
    nvjpegImage.channel[1] = (uint8_t *)(img.gpu_data) + 1 * img.h() * img.w();
    nvjpegImage.channel[2] = (uint8_t *)(img.gpu_data) + 2 * img.h() * img.w();
    nvjpegInputFormat_t nvjpegInputFormat =
        img.colorType == COLOR_TYPE_BGR888_PLANAR ? NVJPEG_INPUT_BGR
                                                  : NVJPEG_INPUT_RGB;
    NVJPEGCHECK(nvjpegEncodeImage(handle_, enc_state_, enc_params_,
                                  &nvjpegImage, nvjpegInputFormat, img.w(),
                                  img.h(), stream_));
    // get compressed stream size
    size_t length;
    NVJPEGCHECK(nvjpegEncodeRetrieveBitstream(handle_, enc_state_, nullptr,
                                              &length, stream_));
    std::vector<char> img_data(length);
    NVJPEGCHECK(nvjpegEncodeRetrieveBitstream(
        handle_, enc_state_, reinterpret_cast<uint8_t *>(img_data.data()),
        &length, nullptr));
    CUDACHECK(cudaStreamSynchronize(stream_));
    LOG_INFO("Jpeg encode image size: {} Bytes", length);
    return img_data;
}
}  // namespace nv