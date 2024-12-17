/***
 * @Author: xingwg
 * @Date: 2024-10-11 11:46:51
 * @LastEditTime: 2024-10-12 09:11:30
 * @FilePath: /dmnn2/src/parsers/caffe/caffeParser/opParsers/parseTanH.cpp
 * @Description:
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#include "opParsers.h"

using namespace nvinfer1;

namespace nvcaffeparser1 {
ILayer *parseTanH(INetworkDefinition &network,
                  const trtcaffe::LayerParameter &msg,
                  CaffeWeightFactory & /*weightFactory*/,
                  BlobNameToTensor &tensors) {
    if (!checkBlobs(msg, 1, 1))
        return nullptr;
    return network.addActivation(*tensors[msg.bottom(0)],
                                 ActivationType::kTANH);
}
} // namespace nvcaffeparser1