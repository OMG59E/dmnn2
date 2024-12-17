/***
 * @Author: xingwg
 * @Date: 2024-10-11 11:46:51
 * @LastEditTime: 2024-10-12 09:11:50
 * @FilePath: /dmnn2/src/parsers/caffe/caffeParser/opParsers/parseUpsample.cpp
 * @Description:
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#include "opParsers.h"

using namespace nvinfer1;

namespace nvcaffeparser1 {
ILayer *parseUpsample(INetworkDefinition &network,
                      const trtcaffe::LayerParameter &msg,
                      CaffeWeightFactory & /*weightFactory*/,
                      BlobNameToTensor &tensors) {
    if (!checkBlobs(msg, 1, 1))
        return nullptr;

    const trtcaffe::UpsampleParameter &p = msg.upsample_param();

    nv::DimsCHW dims =
        parserutils::getCHW(tensors[msg.bottom(0)]->getDimensions());
    int in_h = dims.h();
    int in_w = dims.w();

    int out_h = in_h * p.scale();
    int out_w = in_w * p.scale();

    auto *layer = network.addResize(*tensors[msg.bottom(0)]);
    layer->setName(msg.name().c_str());
    layer->setResizeMode(ResizeMode::kNEAREST);
    layer->setOutputDimensions(Dims3{dims.c(), out_h, out_w});
    return layer;
}
} // namespace nvcaffeparser1
