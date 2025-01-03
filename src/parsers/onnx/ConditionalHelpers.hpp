/***
 * @Author: xingwg
 * @Date: 2024-12-12 15:35:50
 * @LastEditTime: 2024-12-13 10:15:13
 * @FilePath: /dmnn2/src/parsers/onnx/ConditionalHelpers.hpp
 * @Description:
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#pragma once

#include "ImporterContext.hpp"
#include "Status.hpp"
#include <NvInfer.h>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace onnx2trt {

// Given a subgraph, find all of its external inputs (tensors entering the
// subgraph). The result is returned in `subgraphInputs`, which is a map indexed
// by ITensor (a tensor entering the subgraph) and with values indicating a set
// of external input indices.
Status getSubgraphInputs(
    std::vector<nvinfer1::ILayer *> const &newLayers,
    std::unordered_map<nvinfer1::ITensor *, std::set<int32_t>> &subgraphInputs);

// Given a subgraph, find all of its external outputs (tensors exiting the
// subgraph). The result is returned in `subgraphInputs`, which is a map indexed
// by ITensor (a tensor exiting the subgraph) and with values indicating a set
// of external outputs indices.
Status getSubgraphOutputs(
    const std::vector<nvinfer1::ILayer *> &newLayers,
    std::unordered_map<nvinfer1::ITensor *, std::set<int32_t>> &subgraphOutputs,
    const std::vector<std::string> &reportedOutputs);

// Take a snapshot of the network before and after parsing the subgraph and
// return a list of newly added network layers.
Status importSubgraph(IImporterContext *ctx,
                      onnx::GraphProto const &subgraph,
                      std::vector<nvinfer1::ILayer *> &newLayers,
                      StringMap<TensorOrWeights> &subgraphTensors);

using InputsMap =
    std::unordered_map<std::string, nvinfer1::IIfConditionalInputLayer *>;

// Add IIfConditionalInputLayers to the inputs of the subgraph indicated by
// `subgraph`.
onnx2trt::Status
addIfInputLayers(IImporterContext *ctx, nvinfer1::IIfConditional *conditional,
                 InputsMap &inputsMap,
                 const std::vector<nvinfer1::ILayer *> &newLayers);

// Add IIfConditionalOutputLayers to the outputs of the subgraph indicated by
// `subgraph`.
onnx2trt::Status
addIfOutputLayers(IImporterContext *ctx, nvinfer1::IIfConditional *conditional,
                  onnx::GraphProto const &thenGraph,
                  std::vector<nvinfer1::ILayer *> const &thenLayers,
                  StringMap<TensorOrWeights> const &thenSubgraphTensors,
                  onnx::GraphProto const &elseGraph,
                  std::vector<nvinfer1::ILayer *> const &elseLayers,
                  StringMap<TensorOrWeights> const &elseSubgraphTensors,
                  std::vector<TensorOrWeights> &graphOutputs);

}  // namespace onnx2trt