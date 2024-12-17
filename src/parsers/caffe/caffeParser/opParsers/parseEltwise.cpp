/***
 * @Author: xingwg
 * @Date: 2024-10-11 11:46:51
 * @LastEditTime: 2024-10-12 08:50:06
 * @FilePath: /dmnn2/src/parsers/caffe/caffeParser/opParsers/parseEltwise.cpp
 * @Description:
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#include "opParsers.h"

using namespace nvinfer1;

namespace nvcaffeparser1 {
ILayer *parseEltwise(INetworkDefinition &network,
                     const trtcaffe::LayerParameter &msg,
                     CaffeWeightFactory & /*weightFactory*/,
                     BlobNameToTensor &tensors) {
    if (!checkBlobs(msg, 2, 1))
        return nullptr;

    const trtcaffe::EltwiseParameter &p = msg.eltwise_param();

    ElementWiseOperation op = ElementWiseOperation::kSUM;
    switch (p.operation()) {
    case trtcaffe::EltwiseParameter_EltwiseOp_SUM:
        op = ElementWiseOperation::kSUM;
        break;
    case trtcaffe::EltwiseParameter_EltwiseOp_PROD:
        op = ElementWiseOperation::kPROD;
        break;
    case trtcaffe::EltwiseParameter_EltwiseOp_MAX:
        op = ElementWiseOperation::kMAX;
        break;
    }

    return network.addElementWise(*tensors[msg.bottom(0)],
                                  *tensors[msg.bottom(1)], op);
}
} // namespace nvcaffeparser1