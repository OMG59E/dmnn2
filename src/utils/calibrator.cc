/*
 * @Author: xingwg
 * @Date: 2024-10-12 15:52:36
 * @LastEditTime: 2024-12-13 13:46:41
 * @FilePath: /dmnn2/src/utils/calibrator.cc
 * @Description:
 *
 * Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#include "calibrator.h"
#include "convert.h"
#include "error_check.h"
#include <cuda_runtime_api.h>
#include <fstream>
#include <iterator>
#include <random>
#include <string.h>

namespace nv {
BatchStream::BatchStream(const std::string &calibration_data,
                         nvinfer1::INetworkDefinition const *network,
                         int batch_size, int max_batch_idx)
    : calibration_data_(calibration_data), batch_size_(batch_size),
      max_batch_idx_(max_batch_idx) {
    for (int i = 0; i < network->getNbInputs(); ++i) {
        auto input = network->getInput(i);
        nv::Tensor tensor;
        tensor.name = input->getName();
        tensor.nbDims = input->getDimensions().nbDims;
        auto dataType = input->getType();
        tensor.dataType = TrtDataTypeToDmnnDataType(dataType);
        memcpy(tensor.dims, input->getDimensions().d,
               tensor.nbDims * sizeof(int32_t));
        CUDACHECK(cudaMalloc(&tensor.data, tensor.size_bytes()));
        inputs_.insert(std::make_pair(tensor.name, tensor));

        if (!use_rand_data_) {
            std::string filename =
                calibration_data + "/" + tensor.name + "_batch0";
            std::fstream f(filename, std::ios::binary);
            if (!f.is_open()) {
                LOG_WARNING("Failed to open file: {}, will use random data",
                            filename);
                use_rand_data_ = true;
            }
        }
    }
}

BatchStream::~BatchStream() {
    for (auto &tensor : inputs_)
        CUDACHECK(cudaFree(tensor.second.data));
}

bool BatchStream::getBatch(void *bindings[], const char *names[],
                           int nbBindings) {
    if (current_batch_idx_ >= max_batch_idx_)
        return false;

    if (use_rand_data_) {
        gen_random_data(bindings, names, nbBindings);
    } else {
        for (int i = 0; i < nbBindings; ++i) {
            std::string name = names[i];
            const nv::Tensor &tensor = inputs_[name];
            std::string filename = calibration_data_ + "/" + tensor.name +
                                   "_batch" +
                                   std::to_string(current_batch_idx_);
            std::fstream file(filename, std::ios::binary | std::ios::ate);
            if (!file.is_open()) {
                LOG_FATAL("Failed to open file: {}", filename);
            }
            std::streamsize file_size = file.tellg();
            file.seekg(0, std::ios::beg);  // 定位到文件开头
            std::vector<char> buffer(file_size);
            if (!file.read(buffer.data(), file_size)) {
                LOG_FATAL("Failed to read file: {}", filename);
            }
            LOG_ASSERT(tensor.size_bytes() == file_size);
            CUDACHECK(cudaMemcpy(tensor.data, buffer.data(),
                                 tensor.size_bytes(), cudaMemcpyHostToDevice));
            bindings[i] = tensor.data;
        }
    }
    current_batch_idx_++;
    return true;
}
void BatchStream::gen_random_data(void *bindings[], const char *names[],
                                  int nbBindings) {
    std::default_random_engine generator;
    for (int i = 0; i < nbBindings; ++i) {
        std::string name = names[i];
        const nv::Tensor &tensor = inputs_[name];
        if (tensor.dataType == DATA_TYPE_FLOAT32) {
            std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
            auto gen = [&generator, &distribution]() {
                return distribution(generator);
            };
            std::vector<float> rnd_data(tensor.size());
            std::generate_n(rnd_data.begin(), tensor.size(), gen);
            CUDACHECK(cudaMemcpy(tensor.data, rnd_data.data(),
                                 tensor.size_bytes(), cudaMemcpyHostToDevice));
        } else if (tensor.dataType == DATA_TYPE_BOOL) {
            std::uniform_int_distribution<int> distribution(0, 1);
            auto gen = [&distribution, &generator]() {
                return distribution(generator);
            };
            std::vector<int> rnd_data(tensor.size());
            std::generate_n(rnd_data.begin(), tensor.size(), gen);
            CUDACHECK(cudaMemcpy(tensor.data, rnd_data.data(),
                                 tensor.size_bytes(), cudaMemcpyHostToDevice));
        } else if (tensor.dataType == DATA_TYPE_INT32) {
            std::uniform_int_distribution<int32_t> distribution(-255, 255);
            auto gen = [&generator, &distribution]() {
                return distribution(generator);
            };
            std::vector<int32_t> rnd_data(tensor.size());
            std::generate_n(rnd_data.begin(), tensor.size(), gen);
            CUDACHECK(cudaMemcpy(tensor.data, rnd_data.data(),
                                 tensor.size_bytes(), cudaMemcpyHostToDevice));
        } else if (tensor.dataType == DATA_TYPE_UINT32) {
            std::uniform_int_distribution<uint32_t> distribution(0, 255);
            auto gen = [&generator, &distribution]() {
                return distribution(generator);
            };
            std::vector<uint32_t> rnd_data(tensor.size());
            std::generate_n(rnd_data.begin(), tensor.size(), gen);
            CUDACHECK(cudaMemcpy(tensor.data, rnd_data.data(),
                                 tensor.size_bytes(), cudaMemcpyHostToDevice));
        } else if (tensor.dataType == DATA_TYPE_UINT8) {
            std::uniform_int_distribution<uint8_t> distribution(0, 255);
            auto gen = [&generator, &distribution]() {
                return distribution(generator);
            };
            std::vector<uint8_t> rnd_data(tensor.size());
            std::generate_n(rnd_data.begin(), tensor.size(), gen);
            CUDACHECK(cudaMemcpy(tensor.data, rnd_data.data(),
                                 tensor.size_bytes(), cudaMemcpyHostToDevice));
        } else if (tensor.dataType == DATA_TYPE_INT8) {
            std::uniform_int_distribution<int8_t> distribution(-127, 127);
            auto gen = [&generator, &distribution]() {
                return distribution(generator);
            };
            std::vector<int8_t> rnd_data(tensor.size());
            std::generate_n(rnd_data.begin(), tensor.size(), gen);
            CUDACHECK(cudaMemcpy(tensor.data, rnd_data.data(),
                                 tensor.size_bytes(), cudaMemcpyHostToDevice));
        } else if (tensor.dataType == DATA_TYPE_UINT64) {
            std::uniform_int_distribution<uint64_t> distribution(0, 255);
            auto gen = [&generator, &distribution]() {
                return distribution(generator);
            };
            std::vector<uint64_t> rnd_data(tensor.size());
            std::generate_n(rnd_data.begin(), tensor.size(), gen);
            CUDACHECK(cudaMemcpy(tensor.data, rnd_data.data(),
                                 tensor.size_bytes(), cudaMemcpyHostToDevice));
        } else if (tensor.dataType == DATA_TYPE_INT64) {
            std::uniform_int_distribution<int64_t> distribution(-255, 255);
            auto gen = [&generator, &distribution]() {
                return distribution(generator);
            };
            std::vector<int64_t> rnd_data(tensor.size());
            std::generate_n(rnd_data.begin(), tensor.size(), gen);
            CUDACHECK(cudaMemcpy(tensor.data, rnd_data.data(),
                                 tensor.size_bytes(), cudaMemcpyHostToDevice));
        } else {
            LOG_FATAL("Unsupported data type: {}",
                      static_cast<int>(tensor.dataType));
        }
        bindings[i] = tensor.data;
    }
}

//
Int8EntropyCalibrator::Int8EntropyCalibrator(
    const std::string &calibration_data,
    nvinfer1::INetworkDefinition const *network, int batch_size,
    int max_batch_idx, bool read_cache) {
    read_cache_ = read_cache;
    batch_stream_ =
        new BatchStream(calibration_data, network, batch_size, max_batch_idx);
}
Int8EntropyCalibrator::~Int8EntropyCalibrator() { SAFE_FREE(batch_stream_); }
int Int8EntropyCalibrator::getBatchSize() const noexcept {
    return batch_stream_->batch_size_;
}
bool Int8EntropyCalibrator::getBatch(void *bindings[], const char *names[],
                                     int nbBindings) noexcept {
    return batch_stream_->getBatch(bindings, names, nbBindings);
}
const void *
Int8EntropyCalibrator::readCalibrationCache(size_t &length) noexcept {
    calibration_cache_.clear();
    std::ifstream input("calibration.table", std::ios::binary);
    input >> std::noskipws;
    if (read_cache_ && input.good())
        std::copy(std::istream_iterator<char>(input),
                  std::istream_iterator<char>(),
                  std::back_inserter(calibration_cache_));
    length = calibration_cache_.size();
    return length ? &calibration_cache_[0] : nullptr;
}
void Int8EntropyCalibrator::writeCalibrationCache(const void *cache,
                                                  size_t length) noexcept {
    std::ofstream output("calibration.table", std::ios::binary);
    output.write(reinterpret_cast<const char *>(cache), length);
}
}  // namespace nv
