/***
 * @Author: xingwg
 * @Date: 2024-10-21 17:44:57
 * @LastEditTime: 2024-10-22 09:30:35
 * @FilePath: /dmnn2/include/codecs/jpeg_enc.h
 * @Description:
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#pragma once
#include "base_types.h"
#include "error_check.h"
#include <cuda_runtime_api.h>
#include <fstream>
#include <nvjpeg.h>
#include <string>
#include <vector>

namespace nv {
class DECLSPEC_API JpegEncoder {
public:
    JpegEncoder();
    ~JpegEncoder();
    void SetQuality(int quality);
    void SetOptimizedHuffman(bool bUseHuffman);
    void SetSamplingFactors(nvjpegChromaSubsampling_t subsampling);
    std::vector<char> Encode(const nv::Image &img);
    int Encode(const std::string &savePath, const nv::Image &img);

private:
    cudaStream_t stream_;
    nvjpegHandle_t handle_;
    nvjpegEncoderState_t enc_state_;
    nvjpegEncoderParams_t enc_params_;
};
}  // namespace nv