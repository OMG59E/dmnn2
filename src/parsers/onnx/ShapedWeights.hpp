/***
 * @Author: xingwg
 * @Date: 2024-12-12 16:20:38
 * @LastEditTime: 2024-12-12 16:20:51
 * @FilePath: /dmnn2/src/parsers/onnx/ShapedWeights.hpp
 * @Description:
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#pragma once

#include "onnx-ml.pb.h"
#include <NvInfer.h>

namespace onnx2trt {

class ShapedWeights {
public:
    using DataType = int32_t;

    //! Create 1D zero-length ShapedWeights of given type, count()==0, and
    //! values=nullptr.
    static ShapedWeights empty(DataType type);

    //! Construct ShapedWeights that is not expected to be usuable,
    //! except with `operator=` and method `setName()`.
    ShapedWeights() = default;

    explicit ShapedWeights(DataType type, void *values, nvinfer1::Dims shape_);

    size_t count() const;

    size_t size_bytes() const;

    const char *getName() const;

    void setName(const char *name);

    //! True if values exist.
    explicit operator bool() const;

    operator nvinfer1::Weights() const;

    template <typename T> T &at(size_t index) {
        assert(index >= 0 && (index * sizeof(T)) < size_bytes());
        return static_cast<T *>(values)[index];
    }

    template <typename T> const T &at(size_t index) const {
        assert(index >= 0 && (index * sizeof(T)) < size_bytes());
        return static_cast<const T *>(values)[index];
    }

public:
    DataType type{static_cast<DataType>(-1)};
    void *values{nullptr};
    nvinfer1::Dims shape{-1, {}};
    const char *name{};
};

class IImporterContext;

}  // namespace onnx2trt