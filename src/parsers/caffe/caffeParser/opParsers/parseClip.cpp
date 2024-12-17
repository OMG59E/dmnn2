/***
 * @Author: xingwg
 * @Date: 2024-10-11 11:46:51
 * @LastEditTime: 2024-10-12 08:47:31
 * @FilePath: /dmnn2/src/parsers/caffe/caffeParser/opParsers/parseClip.cpp
 * @Description:
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#include "opParsers.h"

using namespace nvinfer1;

namespace nvcaffeparser1 {
ILayer *parseClip(INetworkDefinition &network,
                  const trtcaffe::LayerParameter &msg,
                  CaffeWeightFactory & /* weightFactory */,
                  BlobNameToTensor &tensors) {
    if (!checkBlobs(msg, 1, 1))
        return nullptr;
    const trtcaffe::ClipParameter &p = msg.clip_param();
    float alpha = std::numeric_limits<float>::lowest(); // lower bound
    float beta = std::numeric_limits<float>::max();     // upper bound
    if (p.has_min())
        alpha = p.min();
    if (p.has_max())
        beta = p.max();
    auto layer =
        network.addActivation(*tensors[msg.bottom(0)], ActivationType::kCLIP);
    layer->setAlpha(alpha);
    layer->setBeta(beta);
    return layer;
}
} // namespace nvcaffeparser1