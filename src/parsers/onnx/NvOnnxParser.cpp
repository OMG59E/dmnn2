/*** 
 * @Author: xingwg
 * @Date: 2024-12-13 09:48:53
 * @LastEditTime: 2024-12-13 09:49:04
 * @FilePath: /dmnn2/src/parsers/onnx/NvOnnxParser.cpp
 * @Description: 
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved. 
 */
#include "NvOnnxParser.h"
#include "ModelImporter.hpp"

extern "C" void *createNvOnnxParser_INTERNAL(void *network_, void *logger_,
                                             int version) {
    auto network = static_cast<nvinfer1::INetworkDefinition *>(network_);
    auto logger = static_cast<nvinfer1::ILogger *>(logger_);
    return new onnx2trt::ModelImporter(network, logger);
}

extern "C" int getNvOnnxParserVersion() { return NV_ONNX_PARSER_VERSION; }
