/*
 * @Author: xingwg
 * @Date: 2024-10-12 21:31:59
 * @LastEditTime: 2024-10-17 17:24:41
 * @FilePath: /dmnn2/src/models/net_operator.cc
 * @Description:
 *
 * Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#include "models/net_operator.h"
#include "convert.h"
#include "trt_logger.h"
#include <NvInferPlugin.h>
#include <fstream>

namespace nv {
static TrtLogger gTrtLogger;
static TrtProfiler gTrtProfiler;

int NetOperator::unload() {
    SAFE_FREE(ctx_);
    SAFE_FREE(engine_);
    SAFE_FREE(runtime_);
    CUDACHECK(cudaStreamDestroy(stream_));
    auto tensor_free = [](auto &tensors) {
        for (auto &tensor : tensors)
            tensor.free();
    };
    for (auto &input : inputs_)
        input.free();
    for (auto &output : outputs_)
        output.free();
    return 0;
}

int NetOperator::load(const std::string &engine_file, int device_id) {
    device_id_ = device_id;
    LOG_INFO("Loading model: {}", engine_file);
    if (!initLibNvInferPlugins(&gTrtLogger, "")) {
        LOG_ERROR("Failed to load plugins");
        return -1;
    }
    if (device_id_ < 0) {
        LOG_ERROR("DeviceId must be >= 0");
        return -1;
    }
    CUDACHECK(cudaSetDevice(device_id_));
    CUDACHECK(cudaStreamCreate(&stream_));
    runtime_ = createInferRuntime(gTrtLogger);
    if (!runtime_) {
        LOG_ERROR("Failed to create inference runtime.");
        return -1;
    }
    std::ifstream file(engine_file, std::ios::binary |
                                        std::ios::ate);  // model cache to load
    if (!file.good()) {
        LOG_ERROR("Failed to load engine file.");
        return -1;
    }
    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(file_size);
    if (!file.read(buffer.data(), file_size)) {
        LOG_ERROR("Failed to read engine file.");
        return -1;
    }
    LOG_INFO("Cached model size: {} Bytes", file_size);
    engine_ =
        runtime_->deserializeCudaEngine(buffer.data(), file_size, nullptr);
    if (!engine_) {
        LOG_ERROR("Failed to deserialize engine.");
        return -1;
    }
    ctx_ = engine_->createExecutionContext();
    if (!ctx_) {
        LOG_ERROR("Failed to create execution context.");
        return -1;
    }
    ctx_->setProfiler(&gTrtProfiler);
    int nbBindings = engine_->getNbBindings();
    for (int i = 0; i < nbBindings; ++i) {
        auto name = engine_->getBindingName(i);
        auto dims = engine_->getBindingDimensions(i);
        auto dtype = engine_->getBindingDataType(i);
        bool is_input = engine_->bindingIsInput(i);
        nv::Tensor tensor;
        tensor.idx = i;
        tensor.name = name;
        tensor.nbDims = dims.nbDims;
        memcpy(tensor.dims, dims.d, dims.nbDims * sizeof(int32_t));
        tensor.dataType = TrtDataTypeToDmnnDataType(dtype);
        if (engine_->hasImplicitBatchDimension()) {
            LOG_INFO("Using implicit batch size: {}", dims.d[0]);
            if (is_input && tensor.nbDims != 3)
                LOG_FATAL("Caffemodel input shape must be 4D");
            int max_batch_size = engine_->getMaxBatchSize();
            tensor.nbDims += 1;
            tensor.dims[0] = max_batch_size;
            memcpy(tensor.dims + 1, dims.d, dims.nbDims * sizeof(int32_t));
        }
        CUDACHECK(cudaMalloc(&tensor.gpu_data, tensor.size_bytes()));
        if (is_input) {
            inputs_.emplace_back(tensor);
        } else {
            CUDACHECK(cudaMallocHost(&tensor.data, tensor.size_bytes()));
            outputs_.emplace_back(tensor);
        }
        buffers_[i] = tensor.gpu_data;
    }
    printInputOutputInfo(inputs_, "Input");
    printInputOutputInfo(outputs_, "Output");
    return 0;
}

int NetOperator::inference(int batch_size) {
    if (batch_size <= 0) {
        batch_size = engine_->getMaxBatchSize();
        LOG_INFO("batch_size <= 0 defaults to using the maximum batch({}) for "
                 "inference",
                 batch_size);
    }
    if (engine_->hasImplicitBatchDimension()) {
        if (!ctx_->enqueue(batch_size, buffers_, stream_, nullptr)) {
            LOG_ERROR("Failed to ctx_->enqueue");
            return -1;
        }
    } else {
        for (auto &input : inputs_) {
            auto dims = toTrtDims(input);
            dims.d[0] = batch_size;
            ctx_->setBindingDimensions(input.idx, dims);
        }
        if (!ctx_->enqueueV2(buffers_, stream_, nullptr)) {
            LOG_ERROR("Failed to ctx_->enqueueV2");
            return -1;
        }
    }
    for (auto &output : outputs_)
        CUDACHECK(cudaMemcpyAsync(output.data, output.gpu_data,
                                  output.size_bytes(), cudaMemcpyDeviceToHost,
                                  stream_));
    CUDACHECK(cudaStreamSynchronize(stream_));
    return 0;
}

void NetOperator::printInputOutputInfo(const std::vector<nv::Tensor> &tensors,
                                       const std::string &prefix) const {
    for (int i = 0; i < tensors.size(); ++i) {
        const nv::Tensor &tensor = tensors[i];
        std::string shape_s;
        for (int j = 0; j < tensor.nbDims - 1; ++j) {
            shape_s += std::to_string(tensor.dims[j]) + "x";
        }
        shape_s += std::to_string(tensor.dims[tensor.nbDims - 1]);
        std::string type_s = DmnnDataTypeToString(tensor.dataType);
        LOG_INFO("{}[{}] name: {}, shape: {}, dtype: {}", prefix, i,
                 tensor.name, shape_s, type_s);
    }
}

void NetOperator::printLayerTimes(int iterations) {
    gTrtProfiler.printLayerTimes(iterations);
}
}  // namespace nv