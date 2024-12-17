/***
 * @Author: xingwg
 * @Date: 2024-12-09 15:27:39
 * @LastEditTime: 2024-12-13 11:47:50
 * @FilePath: /dmnn2/src/parsers/caffe/caffeParser/opParsers/opParsers.h
 * @Description:
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#ifndef TRT_CAFFE_PARSER_OP_PARSERS_H
#define TRT_CAFFE_PARSER_OP_PARSERS_H

#include "../../../common/half.h"
#include "../../../common/parserUtils.h"
#include "../../blobNameToTensor.h"
#include "../../caffeWeightFactory/caffeWeightFactory.h"
#include "error_check.h"
#include "trtcaffe.pb.h"
#include <NvInfer.h>
#include <iostream>
#include <unordered_map>

using namespace nvinfer1;

namespace nvcaffeparser1 {
inline bool checkBlobs(const trtcaffe::LayerParameter &msg, int bottoms,
                       int tops) {
    if (msg.bottom_size() != bottoms) {
        LOG_INFO("{}: expected {} bottom blobs, found {}", msg.name(), bottoms,
                 msg.bottom_size());
        return false;
    }

    if (msg.top_size() != tops) {
        LOG_INFO("{}: expected {} tops blobs, found {}", msg.name(), tops,
                 msg.top_size());
        return false;
    }
    return true;
}

typedef ILayer *(*LayerParseFn)(INetworkDefinition &,
                                const trtcaffe::LayerParameter &,
                                CaffeWeightFactory &, BlobNameToTensor &);

ILayer *parseAbsVal(INetworkDefinition &network,
                    const trtcaffe::LayerParameter &msg, CaffeWeightFactory &,
                    BlobNameToTensor &tensors);
ILayer *parseBatchNormalization(INetworkDefinition &network,
                                const trtcaffe::LayerParameter &msg,
                                CaffeWeightFactory &weightFactory,
                                BlobNameToTensor &tensors);
ILayer *parseBNLL(INetworkDefinition &network,
                  const trtcaffe::LayerParameter &msg, CaffeWeightFactory &,
                  BlobNameToTensor &tensors);
ILayer *parseClip(INetworkDefinition &network,
                  const trtcaffe::LayerParameter &msg, CaffeWeightFactory &,
                  BlobNameToTensor &tensors);
ILayer *parseConcat(INetworkDefinition &network,
                    const trtcaffe::LayerParameter &msg, CaffeWeightFactory &,
                    BlobNameToTensor &tensors);
ILayer *parseConvolution(INetworkDefinition &network,
                         const trtcaffe::LayerParameter &msg,
                         CaffeWeightFactory &weightFactory,
                         BlobNameToTensor &tensors);
ILayer *parseCrop(INetworkDefinition &network,
                  const trtcaffe::LayerParameter &msg, CaffeWeightFactory &,
                  BlobNameToTensor &tensors);
ILayer *parseDeconvolution(INetworkDefinition &network,
                           const trtcaffe::LayerParameter &msg,
                           CaffeWeightFactory &weightFactory,
                           BlobNameToTensor &tensors);
ILayer *parseEltwise(INetworkDefinition &network,
                     const trtcaffe::LayerParameter &msg, CaffeWeightFactory &,
                     BlobNameToTensor &tensors);
ILayer *parseELU(INetworkDefinition &network,
                 const trtcaffe::LayerParameter &msg, CaffeWeightFactory &,
                 BlobNameToTensor &tensors);
ILayer *parseInnerProduct(INetworkDefinition &network,
                          const trtcaffe::LayerParameter &msg,
                          CaffeWeightFactory &weightFactory,
                          BlobNameToTensor &tensors);
ILayer *parseLRN(INetworkDefinition &network,
                 const trtcaffe::LayerParameter &msg, CaffeWeightFactory &,
                 BlobNameToTensor &tensors);
ILayer *parsePermute(INetworkDefinition &network,
                     const trtcaffe::LayerParameter &msg, CaffeWeightFactory &,
                     BlobNameToTensor &tensors);
ILayer *parsePooling(INetworkDefinition &network,
                     const trtcaffe::LayerParameter &msg, CaffeWeightFactory &,
                     BlobNameToTensor &tensors);
ILayer *parsePower(INetworkDefinition &network,
                   const trtcaffe::LayerParameter &msg,
                   CaffeWeightFactory &weightFactory,
                   BlobNameToTensor &tensors);
ILayer *parsePReLU(INetworkDefinition &network,
                   const trtcaffe::LayerParameter &msg,
                   CaffeWeightFactory &weightFactory,
                   BlobNameToTensor &tensors);
ILayer *parseReduction(INetworkDefinition &network,
                       const trtcaffe::LayerParameter &msg,
                       CaffeWeightFactory &weightFactory,
                       BlobNameToTensor &tensors);
ILayer *parseReLU(INetworkDefinition &network,
                  const trtcaffe::LayerParameter &msg,
                  CaffeWeightFactory & /*weightFactory*/,
                  BlobNameToTensor &tensors);
ILayer *parseReshape(INetworkDefinition &network,
                     const trtcaffe::LayerParameter &msg,
                     CaffeWeightFactory & /*weightFactory*/,
                     BlobNameToTensor &tensors);
ILayer *parseScale(INetworkDefinition &network,
                   const trtcaffe::LayerParameter &msg,
                   CaffeWeightFactory &weightFactory,
                   BlobNameToTensor &tensors);
ILayer *parseSigmoid(INetworkDefinition &network,
                     const trtcaffe::LayerParameter &msg,
                     CaffeWeightFactory & /*weightFactory*/,
                     BlobNameToTensor &tensors);
ILayer *parseSoftMax(INetworkDefinition &network,
                     const trtcaffe::LayerParameter &msg,
                     CaffeWeightFactory & /*weightFactory*/,
                     BlobNameToTensor &tensors);
ILayer *parseTanH(INetworkDefinition &network,
                  const trtcaffe::LayerParameter &msg,
                  CaffeWeightFactory & /*weightFactory*/,
                  BlobNameToTensor &tensors);
ILayer *parseHardSwish(INetworkDefinition &network,
                       const trtcaffe::LayerParameter &msg,
                       CaffeWeightFactory & /*weightFactory*/,
                       BlobNameToTensor &tensors);
ILayer *parseUpsample(INetworkDefinition &network,
                      const trtcaffe::LayerParameter &msg,
                      CaffeWeightFactory & /*weightFactory*/,
                      BlobNameToTensor &tensors);
ILayer *parseInstanceNorm(INetworkDefinition &network,
                          const trtcaffe::LayerParameter &msg,
                          CaffeWeightFactory &weightFactory,
                          BlobNameToTensor &tensors);
ILayer *parseInterp(INetworkDefinition &network,
                    const trtcaffe::LayerParameter &msg,
                    CaffeWeightFactory & /*weightFactory*/,
                    BlobNameToTensor &tensors);
ILayer *parseFlatten(INetworkDefinition &network,
                     const trtcaffe::LayerParameter &msg,
                     CaffeWeightFactory & /*weightFactory*/,
                     BlobNameToTensor &tensors);

static std::unordered_map<std::string, LayerParseFn> gParseTable{
    {"Convolution", parseConvolution},
    {"Pooling", parsePooling},
    {"InnerProduct", parseInnerProduct},
    {"ReLU", parseReLU},
    {"Softmax", parseSoftMax},
    {"SoftmaxWithLoss", parseSoftMax},
    {"LRN", parseLRN},
    {"Power", parsePower},
    {"Eltwise", parseEltwise},
    {"Concat", parseConcat},
    {"Deconvolution", parseDeconvolution},
    {"Sigmoid", parseSigmoid},
    {"TanH", parseTanH},
    {"BatchNorm", parseBatchNormalization},
    {"Scale", parseScale},
    {"Crop", parseCrop},
    {"Reduction", parseReduction},
    {"Reshape", parseReshape},
    {"Permute", parsePermute},
    {"ELU", parseELU},
    {"BNLL", parseBNLL},
    {"Clip", parseClip},
    {"AbsVal", parseAbsVal},
    {"PReLU", parsePReLU},
    {"HardSwish", parseHardSwish},
    {"Upsample", parseUpsample},
    {"InstanceNorm", parseInstanceNorm},
    {"Interp", parseInterp},
    {"Flatten", parseFlatten},
};
} // namespace nvcaffeparser1

#endif // TRT_CAFFE_PARSER_OP_PARSERS_H
