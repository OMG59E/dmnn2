/***
 * @Author: xingwg
 * @Date: 2024-12-09 15:27:39
 * @LastEditTime: 2024-12-13 12:10:40
 * @FilePath: /dmnn2/src/parsers/caffe/caffeParser/caffeParser.h
 * @Description:
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
/***
 * @Author: xingwg
 * @Date: 2024-10-11 11:46:51
 * @LastEditTime: 2024-10-11 17:37:39
 * @FilePath: /dmnn2/src/parsers/caffe/caffeParser/caffeParser.h
 * @Description:
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */

#ifndef TRT_CAFFE_PARSER_CAFFE_PARSER_H
#define TRT_CAFFE_PARSER_CAFFE_PARSER_H

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <NvCaffeParser.h>

#include "../blobNameToTensor.h"
#include "../caffeWeightFactory/caffeWeightFactory.h"
#include "error_check.h"
#include "trtcaffe.pb.h"

namespace nvcaffeparser1 {
class DECLSPEC_API CaffeParser : public ICaffeParser {
  public:
    IBlobNameToTensor const *
    parse(char const *deploy, char const *model,
          nvinfer1::INetworkDefinition &network,
          nvinfer1::DataType weightType) noexcept override;

    IBlobNameToTensor const *
    parseBuffers(uint8_t const *deployBuffer, size_t deployLength,
                 uint8_t const *modelBuffer, size_t modelLength,
                 nvinfer1::INetworkDefinition &network,
                 nvinfer1::DataType weightType) noexcept override;

    void setProtobufBufferSize(size_t size) noexcept override {
        mProtobufBufferSize = size;
    }
    void setPluginFactoryV2(
        nvcaffeparser1::IPluginFactoryV2 *factory) noexcept override {
        mPluginFactoryV2 = factory;
    }
    void setPluginNamespace(const char *libNamespace) noexcept override {
        mPluginNamespace = libNamespace;
    }
    IBinaryProtoBlob *parseBinaryProto(const char *fileName) noexcept override;
    void destroy() noexcept override { delete this; }
    void
    setErrorRecorder(nvinfer1::IErrorRecorder *recorder) noexcept override {
        (void)recorder;
        LOG_FATAL("TRT- Not implemented.");
    }
    nvinfer1::IErrorRecorder *getErrorRecorder() const noexcept override {
        LOG_FATAL("TRT- Not implemented.");
        return nullptr;
    }

  private:
    ~CaffeParser() override;

    std::vector<nvinfer1::PluginField>
    parseNormalizeParam(const trtcaffe::LayerParameter &msg,
                        CaffeWeightFactory &weightFactory,
                        BlobNameToTensor &tensors);
    std::vector<nvinfer1::PluginField>
    parsePriorBoxParam(const trtcaffe::LayerParameter &msg,
                       CaffeWeightFactory &weightFactory,
                       BlobNameToTensor &tensors);
    std::vector<nvinfer1::PluginField>
    parseDetectionOutputParam(const trtcaffe::LayerParameter &msg,
                              CaffeWeightFactory &weightFactory,
                              BlobNameToTensor &tensors);
    std::vector<nvinfer1::PluginField>
    parseDetectionOutputV2Param(const trtcaffe::LayerParameter &msg,
                                CaffeWeightFactory &weightFactory,
                                BlobNameToTensor &tensors);
    // std::vector<nvinfer1::PluginField> parseLReLUParam(const
    // trtcaffe::LayerParameter& msg, CaffeWeightFactory& weightFactory,
    // BlobNameToTensor& tensors); std::vector<nvinfer1::PluginField>
    // parseRPROIParam(const trtcaffe::LayerParameter& msg, CaffeWeightFactory&
    // weightFactory, BlobNameToTensor& tensors);
    // std::vector<nvinfer1::PluginField> parseFlattenParam(const
    // trtcaffe::LayerParameter& msg, CaffeWeightFactory& weightFactory,
    // BlobNameToTensor& tensors); std::vector<nvinfer1::PluginField>
    // parseInterpParam(const trtcaffe::LayerParameter& msg, CaffeWeightFactory&
    // weightFactory, BlobNameToTensor& tensors);
    std::vector<nvinfer1::PluginField>
    parseSliceParam(const trtcaffe::LayerParameter &msg,
                    CaffeWeightFactory &weightFactory,
                    BlobNameToTensor &tensors);
    std::vector<nvinfer1::PluginField>
    parseScaleV2Param(const trtcaffe::LayerParameter &msg,
                      CaffeWeightFactory &weightFactory,
                      BlobNameToTensor &tensors);
    // std::vector<nvinfer1::PluginField> parseUpsampleParam(const
    // trtcaffe::LayerParameter& msg, CaffeWeightFactory& weightFactory,
    // BlobNameToTensor& tensors);
    std::vector<nvinfer1::PluginField>
    parseYoloBoxParam(const trtcaffe::LayerParameter &msg,
                      CaffeWeightFactory &weightFactory,
                      BlobNameToTensor &tensors);
    std::vector<nvinfer1::PluginField>
    parseYoloDetectionOutputParam(const trtcaffe::LayerParameter &msg,
                                  CaffeWeightFactory &weightFactory,
                                  BlobNameToTensor &tensors);
    // std::vector<nvinfer1::PluginField> parseBroadcastMulParam(const
    // trtcaffe::LayerParameter& msg, CaffeWeightFactory& weightFactory,
    // BlobNameToTensor& tensors); std::vector<nvinfer1::PluginField>
    // parseAddParam(const trtcaffe::LayerParameter& msg, CaffeWeightFactory&
    // weightFactory, BlobNameToTensor& tensors);
    // std::vector<nvinfer1::PluginField> parseDivParam(const
    // trtcaffe::LayerParameter& msg, CaffeWeightFactory& weightFactory,
    // BlobNameToTensor& tensors); std::vector<nvinfer1::PluginField>
    // parseMulParam(const trtcaffe::LayerParameter& msg, CaffeWeightFactory&
    // weightFactory, BlobNameToTensor& tensors);
    // std::vector<nvinfer1::PluginField> parseHardSwishParam(const
    // trtcaffe::LayerParameter& msg, CaffeWeightFactory& weightFactory,
    // BlobNameToTensor& tensors); std::vector<nvinfer1::PluginField>
    // parseInstanceNormParam(const trtcaffe::LayerParameter&
    // msg,CaffeWeightFactory& weightFactory,BlobNameToTensor& tensors);
    std::vector<nvinfer1::PluginField>
    parseCenterFaceOutputParam(const trtcaffe::LayerParameter &msg,
                               CaffeWeightFactory &weightFactory,
                               BlobNameToTensor &tensors);
    std::vector<nvinfer1::PluginField>
    parseYOLOXDetectionOutputParam(const trtcaffe::LayerParameter &msg,
                                   CaffeWeightFactory &weightFactory,
                                   BlobNameToTensor &tensors);
    std::vector<nvinfer1::PluginField>
    parseFocusParam(const trtcaffe::LayerParameter &msg,
                    CaffeWeightFactory &weightFactory,
                    BlobNameToTensor &tensors);

    template <typename T> T *allocMemory(int size = 1) {
        T *tmpMem = static_cast<T *>(malloc(sizeof(T) * size));
        mTmpAllocs.push_back(tmpMem);
        return tmpMem;
    }

    const IBlobNameToTensor *parse(nvinfer1::INetworkDefinition &network,
                                   nvinfer1::DataType weightType,
                                   bool hasModel);

  private:
    std::shared_ptr<trtcaffe::NetParameter> mDeploy;
    std::shared_ptr<trtcaffe::NetParameter> mModel;
    std::vector<void *> mTmpAllocs;
    BlobNameToTensor *mBlobNameToTensor{nullptr};
    size_t mProtobufBufferSize{INT_MAX};
    // nvcaffeparser1::IPluginFactory *mPluginFactory{nullptr};
    nvcaffeparser1::IPluginFactoryV2 *mPluginFactoryV2{nullptr};
    bool mPluginFactoryIsExt{false};
    std::vector<nvinfer1::IPluginV2 *> mNewPlugins;
    std::unordered_map<std::string, nvinfer1::IPluginCreator *> mPluginRegistry;
    std::string mPluginNamespace = "";
};
} // namespace nvcaffeparser1
#endif // TRT_CAFFE_PARSER_CAFFE_PARSER_H
