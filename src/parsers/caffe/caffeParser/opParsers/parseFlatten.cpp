/***
 * @Author: xingwg
 * @Date: 2024-10-11 11:46:51
 * @LastEditTime: 2024-10-12 07:59:36
 * @FilePath: /dmnn2/src/parsers/caffe/caffeParser/opParsers/parseFlatten.cpp
 * @Description:
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#include "opParsers.h"

using namespace nvinfer1;

namespace nvcaffeparser1 {
ILayer *parseFlatten(INetworkDefinition &network,
                     const trtcaffe::LayerParameter &msg,
                     CaffeWeightFactory & /*weightFactory*/,
                     BlobNameToTensor &tensors) {
    if (!checkBlobs(msg, 1, 1))
        return nullptr;

    const trtcaffe::FlattenParameter &p = msg.flatten_param();
    Dims bottomDims = tensors[msg.bottom(0)]->getDimensions();
    if (bottomDims.nbDims != 3) {
        LOG_ERROR("Input tensor dim must be = 3");
        return nullptr;
    }

    Dims topDims;
    topDims.nbDims = bottomDims.nbDims;
    topDims.d[0] = bottomDims.d[0] * bottomDims.d[1] * bottomDims.d[2];
    topDims.d[1] = 1;
    topDims.d[2] = 1;

    auto *layer = network.addShuffle(*tensors[msg.bottom(0)]);
    layer->setReshapeDimensions(topDims);
    return layer;
}
} // namespace nvcaffeparser1