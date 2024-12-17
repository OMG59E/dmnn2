/***
 * @Author: xingwg
 * @Date: 2024-12-12 17:16:20
 * @LastEditTime: 2024-12-12 17:16:34
 * @FilePath: /dmnn2/src/parsers/onnx/OnnxAttrs.hpp
 * @Description:
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#pragma once

#include "onnx-ml.pb.h"
#include <NvInfer.h>
#include <unordered_map>
#include <vector>

#include "ImporterContext.hpp"

class OnnxAttrs {
    template <typename T> using string_map = std::unordered_map<std::string, T>;
    typedef string_map<onnx::AttributeProto const *> AttrMap;
    AttrMap _attrs;
    onnx2trt::IImporterContext *mCtx;

public:
    explicit OnnxAttrs(onnx::NodeProto const &onnx_node,
                       onnx2trt::IImporterContext *ctx)
        : mCtx{ctx} {
        for (auto const &attr : onnx_node.attribute()) {
            _attrs.insert({attr.name(), &attr});
        }
    }

    bool count(const std::string &key) const { return _attrs.count(key); }

    onnx::AttributeProto const *at(std::string key) const {
        if (!_attrs.count(key)) {
            throw std::out_of_range("Attribute not found: " + key);
        }
        return _attrs.at(key);
    }

    onnx::AttributeProto::AttributeType type(const std::string &key) const {
        return this->at(key)->type();
    }

    template <typename T> T get(const std::string &key) const;

    template <typename T>
    T get(const std::string &key, T const &default_value) const {
        return _attrs.count(key) ? this->get<T>(key) : default_value;
    }
};
