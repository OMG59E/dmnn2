/***
 * @Author: xingwg
 * @Date: 2024-10-11 11:46:51
 * @LastEditTime: 2024-10-12 09:10:05
 * @FilePath: /dmnn2/src/parsers/caffe/caffeParser/opParsers/parseReLU.cpp
 * @Description:
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#include "opParsers.h"

using namespace nvinfer1;

namespace nvcaffeparser1 {
ILayer *parseReLU(INetworkDefinition &network,
                  const trtcaffe::LayerParameter &msg,
                  CaffeWeightFactory & /*weightFactory*/,
                  BlobNameToTensor &tensors) {
    if (!checkBlobs(msg, 1, 1))
        return nullptr;

    const trtcaffe::ReLUParameter &p = msg.relu_param();

    if (p.has_negative_slope() && p.negative_slope() != 0) {
        auto newLayer = network.addActivation(*tensors[msg.bottom(0)],
                                              ActivationType::kLEAKY_RELU);
        newLayer->setAlpha(p.negative_slope());
        return newLayer;
    }
    return network.addActivation(*tensors[msg.bottom(0)],
                                 ActivationType::kRELU);
}
} // namespace nvcaffeparser1