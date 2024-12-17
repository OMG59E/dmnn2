/***
 * @Author: xingwg
 * @Date: 2024-12-09 15:27:39
 * @LastEditTime: 2024-12-13 14:03:44
 * @FilePath: /dmnn2/src/utils/convert.h
 * @Description:
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#include "base_types.h"
#include "logging.h"
#include <NvInfer.h>

static nv::DataType TrtDataTypeToDmnnDataType(nvinfer1::DataType dataType) {
    switch (dataType) {
    case nvinfer1::DataType::kFLOAT:
        return nv::DATA_TYPE_FLOAT32;
    case nvinfer1::DataType::kHALF:
        return nv::DATA_TYPE_FLOAT16;
    case nvinfer1::DataType::kINT32:
        return nv::DATA_TYPE_INT32;
    case nvinfer1::DataType::kINT8:
        return nv::DATA_TYPE_INT8;
    case nvinfer1::DataType::kBOOL:
        return nv::DATA_TYPE_BOOL;
    case nvinfer1::DataType::kUINT8:
        return nv::DATA_TYPE_UINT8;
    default:
        LOG_FATAL("Unsupported data type: {}", static_cast<int>(dataType));
        break;
    };
}

static std::string DmnnDataTypeToString(nv::DataType dataType) {
    switch (dataType) {
    case nv::DATA_TYPE_FLOAT32:
        return "float32";
    case nv::DATA_TYPE_FLOAT16:
        return "float16";
    case nv::DATA_TYPE_INT32:
        return "int32";
    case nv::DATA_TYPE_INT8:
        return "int8";
    case nv::DATA_TYPE_BOOL:
        return "bool";
    case nv::DATA_TYPE_UINT8:
        return "uint8";
    default:
        LOG_FATAL("Unsupported data type: {}", static_cast<int>(dataType));
        break;
    };
}

static nvinfer1::Dims toTrtDims(const nv::Tensor &tensor) {
    nvinfer1::Dims dims;
    dims.nbDims = tensor.nbDims;
    for (int i = 0; i < tensor.nbDims; ++i)
        dims.d[i] = tensor.dims[i];
    return dims;
}