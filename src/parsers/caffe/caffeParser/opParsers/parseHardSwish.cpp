/***
 * @Author: xingwg
 * @Date: 2024-10-11 11:46:51
 * @LastEditTime: 2024-10-12 08:50:40
 * @FilePath: /dmnn2/src/parsers/caffe/caffeParser/opParsers/parseHardSwish.cpp
 * @Description:
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#include "opParsers.h"

using namespace nvinfer1;

namespace nvcaffeparser1 {
ILayer *parseHardSwish(INetworkDefinition &network,
                       const trtcaffe::LayerParameter &msg,
                       CaffeWeightFactory &weightFactory,
                       BlobNameToTensor &tensors) {
    if (!checkBlobs(msg, 1, 1))
        return nullptr;

    auto clip =
        network.addActivation(*tensors[msg.bottom(0)], ActivationType::kCLIP);
    clip->setAlpha(-3.0f);
    clip->setBeta(3.0f);

    auto *scale_val = reinterpret_cast<float *>(malloc(sizeof(float)));
    *scale_val = 1.0f / 6.0f;
    Weights scale{DataType::kFLOAT, scale_val, 1};
    weightFactory.getTmpAllocs().push_back(scale_val);

    auto *shift_val = reinterpret_cast<float *>(malloc(sizeof(float)));
    *shift_val = 0.5f;
    Weights shift{DataType::kFLOAT, shift_val, 1};
    weightFactory.getTmpAllocs().push_back(shift_val);

    Weights power = weightFactory.getNullWeights();

    weightFactory.convert(shift);
    weightFactory.convert(scale);
    weightFactory.convert(power);

    auto axpy = network.addScale(*clip->getOutput(0), ScaleMode::kUNIFORM,
                                 shift, scale, power);
    auto hs =
        network.addElementWise(*axpy->getOutput(0), *tensors[msg.bottom(0)],
                               ElementWiseOperation::kPROD);
    return hs;
}
} // namespace nvcaffeparser1
