/***
 * @Author: xingwg
 * @Date: 2024-10-11 11:46:51
 * @LastEditTime: 2024-10-12 08:44:15
 * @FilePath: /dmnn2/src/parsers/caffe/caffeParser/opParsers/parseBNLL.cpp
 * @Description:
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#include "opParsers.h"

using namespace nvinfer1;

namespace nvcaffeparser1 {
ILayer *parseBNLL(INetworkDefinition &network,
                  const trtcaffe::LayerParameter &msg, CaffeWeightFactory &,
                  BlobNameToTensor &tensors) {
    if (!checkBlobs(msg, 1, 1))
        return nullptr;
    return network.addActivation(*tensors[msg.bottom(0)],
                                 ActivationType::kSOFTPLUS);
}
} // namespace nvcaffeparser1