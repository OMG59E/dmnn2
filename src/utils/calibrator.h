/***
 * @Author: xingwg
 * @Date: 2024-10-12 15:53:14
 * @LastEditTime: 2024-10-15 11:40:47
 * @FilePath: /dmnn2/src/utils/calibrator.h
 * @Description:
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#ifndef ALGORITHMS_CALIBRATOR_H
#define ALGORITHMS_CALIBRATOR_H
#include "base_types.h"
#include "error_check.h"
#include <NvInfer.h>
#include <map>
#include <vector>

namespace nv {
class DECLSPEC_API BatchStream {
public:
    BatchStream(const std::string &calibration_data,
                nvinfer1::INetworkDefinition const *network, int batch_size,
                int max_batch_idx);
    ~BatchStream();
    bool getBatch(void *bindings[], const char *names[], int nbBindings);

private:
    void gen_random_data(void *bindings[], const char *names[], int nbBindings);

public:
    int batch_size_{0};
    // private:
    bool use_rand_data_{false};
    int max_batch_idx_{100};
    int current_batch_idx_{0};
    std::map<std::string, nv::Tensor> inputs_;
    std::string calibration_data_;
};

class DECLSPEC_API Int8EntropyCalibrator
    : public nvinfer1::IInt8EntropyCalibrator2 {
public:
    Int8EntropyCalibrator(const std::string &calibration_data,
                          nvinfer1::INetworkDefinition const *network,
                          int batch_size, int max_batch_idx,
                          bool read_cache = true);
    virtual ~Int8EntropyCalibrator() override;
    int getBatchSize() const noexcept override;
    bool getBatch(void *bindings[], const char *names[],
                  int nbBindings) noexcept override;
    const void *readCalibrationCache(size_t &length) noexcept override;
    void writeCalibrationCache(const void *cache,
                               size_t length) noexcept override;

private:
    BatchStream *batch_stream_{nullptr};
    bool read_cache_{true};
    std::vector<char> calibration_cache_;
};
}  // namespace nv

#endif  // ALGORITHMS_CALIBRATOR_H