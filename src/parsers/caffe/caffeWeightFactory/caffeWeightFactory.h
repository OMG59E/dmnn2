/***
 * @Author: xingwg
 * @Date: 2024-10-11 11:46:51
 * @LastEditTime: 2024-10-12 09:20:13
 * @FilePath: /dmnn2/src/parsers/caffe/caffeWeightFactory/caffeWeightFactory.h
 * @Description:
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#ifndef TRT_CAFFE_PARSER_CAFFE_WEIGHT_FACTORY_H
#define TRT_CAFFE_PARSER_CAFFE_WEIGHT_FACTORY_H

#include <memory>
#include <random>
#include <string>
#include <vector>

#include <NvInfer.h>

#include "trtcaffe.pb.h"
#include "weightType.h"

namespace nvcaffeparser1 {
class CaffeWeightFactory {
  public:
    CaffeWeightFactory(const trtcaffe::NetParameter &msg,
                       nvinfer1::DataType dataType,
                       std::vector<void *> &tmpAllocs, bool isInitialized);

    nvinfer1::DataType getDataType() const;

    size_t getDataTypeSize() const;

    std::vector<void *> &getTmpAllocs();

    int getBlobsSize(const std::string &layerName);

    const trtcaffe::BlobProto *getBlob(const std::string &layerName, int index);

    std::vector<nvinfer1::Weights> getAllWeights(const std::string &layerName);

    virtual nvinfer1::Weights operator()(const std::string &layerName,
                                         WeightType weightType);

    void convert(nvinfer1::Weights &weights, nvinfer1::DataType targetType);

    void convert(nvinfer1::Weights &weights);

    bool isOK();

    bool isInitialized();

    nvinfer1::Weights getNullWeights();

    nvinfer1::Weights
    allocateWeights(int64_t elems,
                    std::uniform_real_distribution<float> distribution =
                        std::uniform_real_distribution<float>(-0.01f, 0.01F));

    nvinfer1::Weights
    allocateWeights(int64_t elems,
                    std::normal_distribution<float> distribution);

    static trtcaffe::Type
    getBlobProtoDataType(const trtcaffe::BlobProto &blobMsg);

    static size_t sizeOfCaffeType(trtcaffe::Type type);

    // The size returned here is the number of array entries, not bytes
    static std::pair<const void *, size_t>
    getBlobProtoData(const trtcaffe::BlobProto &blobMsg, trtcaffe::Type type,
                     std::vector<void *> &tmpAllocs);

  private:
    template <typename T>
    bool checkForNans(const void *values, int count,
                      const std::string &layerName);

    nvinfer1::Weights getWeights(const trtcaffe::BlobProto &blobMsg,
                                 const std::string &layerName);

    const trtcaffe::NetParameter &mMsg;
    std::unique_ptr<trtcaffe::NetParameter> mRef;
    std::vector<void *> &mTmpAllocs;
    nvinfer1::DataType mDataType;
    // bool mQuantize;
    bool mInitialized;
    std::default_random_engine generator;
    bool mOK{true};
};
} // namespace nvcaffeparser1

#endif // TRT_CAFFE_PARSER_CAFFE_WEIGHT_FACTORY_H