/***
 * @Author: xingwg
 * @Date: 2024-10-15 13:50:12
 * @LastEditTime: 2024-10-21 09:41:28
 * @FilePath: /dmnn2/include/models/net_operator.h
 * @Description:
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#pragma once
#include "base_types.h"
#include <NvInfer.h>
#include <NvInferVersion.h>

#define MAX_INPUT_OUTPUT 128

namespace nv {
class DECLSPEC_API NetOperator {
public:
    NetOperator() = default;
    ~NetOperator() = default;
    virtual int load(const std::string &model_file, int device_id);
    virtual int inference(int batch_size = -1);
    virtual int unload();
    virtual std::string getInputName(int idx) { return inputs_[idx].name; }
    virtual std::string getOutputName(int idx) { return outputs_[idx].name; }
    virtual int getNbOutputs() const { return outputs_.size(); }
    virtual int getNbInputs() const { return inputs_.size(); }
    virtual std::vector<nv::Tensor> &getInputs() { return inputs_; }
    virtual std::vector<nv::Tensor> &getOutputs() { return outputs_; }
    virtual void printLayerTimes(int iterations);
    virtual void printInputOutputInfo(const std::vector<nv::Tensor> &tensors,
                                      const std::string &prefix) const;

public:
    int valid_batch_size_;
    int device_id_;
    void *buffers_[MAX_INPUT_OUTPUT]{};
    std::vector<nv::Tensor> inputs_;
    std::vector<nv::Tensor> outputs_;
    cudaStream_t stream_;
    nvinfer1::IRuntime *runtime_{nullptr};
    nvinfer1::ICudaEngine *engine_{nullptr};
    nvinfer1::IExecutionContext *ctx_{nullptr};
};
}  // namespace nv