/***
 * @Author: xingwg
 * @Date: 2024-12-13 10:17:26
 * @LastEditTime: 2024-12-13 10:17:36
 * @FilePath: /dmnn2/src/parsers/onnx/builtin_op_importers.hpp
 * @Description:
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#pragma once

#include "onnx2trt.hpp"
#include "utils.hpp"

namespace onnx2trt {

string_map<NodeImporter> &getBuiltinOpImporterMap();

} // namespace onnx2trt