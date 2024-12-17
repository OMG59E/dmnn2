/***
 * @Author: xingwg
 * @Date: 2024-12-13 11:38:47
 * @LastEditTime: 2024-12-13 14:34:16
 * @FilePath: /dmnn2/src/parsers/onnx/custom_type_format.h
 * @Description:
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#pragma once

#include "ShapeTensor.hpp"
#include "logging.h"
#include <NvInfer.h>
#include <sstream>

// 定义自定义格式化函数
template <>
struct fmt::formatter<onnx2trt::ShapeTensor> : fmt::formatter<std::string> {
    template <typename FormatContext>
    auto format(const onnx2trt::ShapeTensor &p, FormatContext &ctx) {
        std::stringstream ss{};
        ss << p;
        return fmt::format_to(ctx.out(), ss.str());
    }
};

// 定义自定义格式化函数
template <>
struct fmt::formatter<nvinfer1::DataType> : fmt::formatter<std::string> {
    template <typename FormatContext>
    auto format(const nvinfer1::DataType &dataType, FormatContext &ctx) {
        std::string dtype_s;
        switch (dataType) {
        case nvinfer1::DataType::kFLOAT:
            dtype_s = "float32";
            break;
        case nvinfer1::DataType::kHALF:
            dtype_s = "float16";
            break;
        case nvinfer1::DataType::kINT32:
            dtype_s = "int32";
            break;
        case nvinfer1::DataType::kINT8:
            dtype_s = "int8";
            break;
        case nvinfer1::DataType::kBOOL:
            dtype_s = "bool";
            break;
        case nvinfer1::DataType::kUINT8:
            dtype_s = "uint8";
            break;
        case nvinfer1::DataType::kFP8:
            dtype_s = "float8";
            break;
        default:
            LOG_FATAL("Unsupported data type: {}", static_cast<int>(dataType));
            break;
        };
        return fmt::format_to(ctx.out(), dtype_s);
    }
};

template <>
struct fmt::formatter<nvinfer1::Dims> : fmt::formatter<std::string> {
    template <typename FormatContext>
    auto format(const nvinfer1::Dims &p, FormatContext &ctx) {
        std::string shape_s;
        for (int n = 0; n < p.nbDims - 1; ++n) {
            shape_s += std::to_string(p.d[n]) + " ";
        }
        shape_s += std::to_string(p.d[p.nbDims - 1]);
        return fmt::format_to(ctx.out(), shape_s);
    }
};