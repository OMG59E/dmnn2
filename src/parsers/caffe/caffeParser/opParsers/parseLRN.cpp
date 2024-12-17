/***
 * @Author: xingwg
 * @Date: 2024-10-11 11:46:51
 * @LastEditTime: 2024-10-12 08:54:56
 * @FilePath: /dmnn2/src/parsers/caffe/caffeParser/opParsers/parseLRN.cpp
 * @Description:
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#include "opParsers.h"

using namespace nvinfer1;

namespace nvcaffeparser1 {
ILayer *parseLRN(INetworkDefinition &network,
                 const trtcaffe::LayerParameter &msg,
                 CaffeWeightFactory & /*weightFactory*/,
                 BlobNameToTensor &tensors) {
    if (!checkBlobs(msg, 1, 1))
        return nullptr;

    const trtcaffe::LRNParameter &p = msg.lrn_param();
    int localSize = p.has_local_size() ? p.local_size() : 5;
    float alpha = p.has_alpha() ? p.alpha() : 1;
    float beta = p.has_beta() ? p.beta() : 5;
    float k = p.has_k() ? p.k() : 1;

    return network.addLRN(*tensors[msg.bottom(0)], localSize, alpha, beta, k);
}
} // namespace nvcaffeparser1