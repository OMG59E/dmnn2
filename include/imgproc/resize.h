/***
 * @Author: xingwg
 * @Date: 2024-12-09 15:29:02
 * @LastEditTime: 2024-12-13 16:10:02
 * @FilePath: /dmnn2/include/imgproc/resize.h
 * @Description:
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#pragma once
#include "base_types.h"
#include "logging.h"

namespace nv {
int resize(const nv::Image &src, nv::Image &dst);
int resizePaddingCvtColor(const nv::Image &src, nv::Image &dst,
                          PaddingMode padding_mode, float *padding_values);
int resizePaddingCvtColorAsync(cudaStream_t stream, const nv::Image &src,
                               nv::Image &dst, PaddingMode padding_mode,
                               float *padding_values);
int resizePaddingCvtColorNorm(const nv::Image &src, nv::Image &dst,
                              float *mean_vals, float *std_vals,
                              PaddingMode padding_mode, float *padding_values);
int resizePaddingCvtColorNormAsync(cudaStream_t stream, const nv::Image &src,
                                   nv::Image &dst, float *mean_vals,
                                   float *std_vals, PaddingMode padding_mode,
                                   float *padding_values);
}  // namespace nv