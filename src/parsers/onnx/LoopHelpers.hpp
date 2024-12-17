/***
 * @Author: xingwg
 * @Date: 2024-12-13 10:01:15
 * @LastEditTime: 2024-12-13 10:01:27
 * @FilePath: /dmnn2/src/parsers/onnx/LoopHelpers.hpp
 * @Description:
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#pragma once

#include <NvInfer.h>

#include "ImporterContext.hpp"

namespace onnx2trt {

nvinfer1::ITensor *addLoopCounter(IImporterContext *ctx, nvinfer1::ILoop *loop,
                                  int32_t initial = 0);

} // namespace onnx2trt