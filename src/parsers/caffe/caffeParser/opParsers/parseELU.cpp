/***
 * @Author: xingwg
 * @Date: 2024-10-11 11:46:51
 * @LastEditTime: 2024-10-12 08:50:26
 * @FilePath: /dmnn2/src/parsers/caffe/caffeParser/opParsers/parseELU.cpp
 * @Description:
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#include "opParsers.h"

using namespace nvinfer1;

namespace nvcaffeparser1 {
ILayer *parseELU(INetworkDefinition &network,
                 const trtcaffe::LayerParameter &msg,
                 CaffeWeightFactory & /* weightFactory */,
                 BlobNameToTensor &tensors) {
    if (!checkBlobs(msg, 1, 1))
        return nullptr;

    const trtcaffe::ELUParameter &p = msg.elu_param();

    float alpha = 1.f; // default parameter
    if (p.has_alpha())
        alpha = p.alpha();
    auto newLayer =
        network.addActivation(*tensors[msg.bottom(0)], ActivationType::kELU);
    newLayer->setAlpha(alpha);
    return newLayer;
}
} // namespace nvcaffeparser1