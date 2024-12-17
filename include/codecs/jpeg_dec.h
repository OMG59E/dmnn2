/***
 * @Author: xingwg
 * @Date: 2024-10-21 16:13:47
 * @LastEditTime: 2024-10-22 14:06:50
 * @FilePath: /dmnn2/include/codecs/jpeg_dec.h
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
class DECLSPEC_API JpegDecoder {
public:
    JpegDecoder();
    ~JpegDecoder();
    int Decode(const std::string &filename, nv::Image &image);
    int Decode(const std::vector<char> &data, nv::Image &image);

private:
    cudaStream_t stream_;
    nvjpegHandle_t handle_;
    nvjpegJpegState_t jpeg_state_;
};
}  // namespace nv