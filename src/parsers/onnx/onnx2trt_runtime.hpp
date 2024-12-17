/***
 * @Author: xingwg
 * @Date: 2024-12-12 17:44:47
 * @LastEditTime: 2024-12-13 14:36:11
 * @FilePath: /dmnn2/src/parsers/onnx/onnx2trt_runtime.hpp
 * @Description:
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#pragma once

#include "onnx2trt_common.hpp"

namespace onnx2trt {

typedef Plugin *(*plugin_deserializer)(const void *serialData,
                                       size_t serialLength);

}  // namespace onnx2trt