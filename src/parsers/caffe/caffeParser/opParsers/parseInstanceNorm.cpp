/***
 * @Author: xingwg
 * @Date: 2024-10-11 11:46:51
 * @LastEditTime: 2024-10-12 08:39:07
 * @FilePath:
 * /dmnn2/src/parsers/caffe/caffeParser/opParsers/parseInstanceNorm.cpp
 * @Description:
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#include "opParsers.h"

using namespace nvinfer1;

namespace nvcaffeparser1 {
ILayer *parseInstanceNorm(INetworkDefinition &network,
                          const trtcaffe::LayerParameter &msg,
                          CaffeWeightFactory &weightFactory,
                          BlobNameToTensor &tensors) {
    if (!checkBlobs(msg, 1, 1))
        return nullptr;

    const trtcaffe::InstanceNormParameter &p = msg.instance_normalize_param();

    const float eps = p.eps();
    std::vector<Weights> weights;
    weights = weightFactory.getAllWeights(msg.name());
    Weights scales = weights[0];
    Weights bias = weights[1];
    weightFactory.convert(scales);
    weightFactory.convert(bias);

    // C,H,W -> C,1,1   m = mean(x)
    auto *layer_reduce1 = network.addReduce(*tensors[msg.bottom(0)],
                                            ReduceOperation::kAVG, 6, true);
    if (!layer_reduce1)
        LOG_FATAL("Failed to create reduce layer");
    // C,H,W  C,1,1 -> C,H,W   (x - m)
    auto layer_elwise1 = network.addElementWise(*tensors[msg.bottom(0)],
                                                *layer_reduce1->getOutput(0),
                                                ElementWiseOperation::kSUB);
    if (!layer_elwise1)
        LOG_FATAL("Failed to create elwise layer");

    auto *shift11 = reinterpret_cast<float *>(malloc(sizeof(float)));
    *shift11 = 0.0f;
    auto *scale11 = reinterpret_cast<float *>(malloc(sizeof(float)));
    *scale11 = 1.0f;
    auto *power11 = reinterpret_cast<float *>(malloc(sizeof(float)));
    *power11 = 2.0f;
    Weights shift1{DataType::kFLOAT, shift11, 1};
    Weights scale1{DataType::kFLOAT, scale11, 1};
    Weights power1{DataType::kFLOAT, power11, 1};
    weightFactory.convert(shift1);
    weightFactory.convert(scale1);
    weightFactory.convert(power1);
    weightFactory.getTmpAllocs().push_back(shift11);
    weightFactory.getTmpAllocs().push_back(scale11);
    weightFactory.getTmpAllocs().push_back(power11);

    // C,H,W -> C,H,W   (x - m)^2
    auto *layer_scale1 =
        network.addScale(*layer_elwise1->getOutput(0), ScaleMode::kUNIFORM,
                         shift1, scale1, power1);
    if (!layer_scale1)
        LOG_FATAL("Failed to create scale layer");

    // C,H,W -> C,1,1  s = mean((x - m)^2)
    auto *layer_reduce2 = network.addReduce(*layer_scale1->getOutput(0),
                                            ReduceOperation::kAVG, 6, true);
    if (!layer_reduce2)
        LOG_FATAL("Failed to create reduce layer");

    auto *shift21 = reinterpret_cast<float *>(malloc(sizeof(float)));
    *shift21 = eps;
    auto *scale21 = reinterpret_cast<float *>(malloc(sizeof(float)));
    *scale21 = 1.0f;
    auto *power21 = reinterpret_cast<float *>(malloc(sizeof(float)));
    *power21 = 0.5f;
    Weights shift2{DataType::kFLOAT, shift21, 1};
    Weights scale2{DataType::kFLOAT, scale21, 1};
    Weights power2{DataType::kFLOAT, power21, 1};
    weightFactory.convert(shift2);
    weightFactory.convert(scale2);
    weightFactory.convert(power2);
    weightFactory.getTmpAllocs().push_back(shift21);
    weightFactory.getTmpAllocs().push_back(scale21);
    weightFactory.getTmpAllocs().push_back(power21);
    // C,1,1 -> C,1,1   (s + eps)^0.5
    auto *layer_scale2 =
        network.addScale(*layer_reduce2->getOutput(0), ScaleMode::kUNIFORM,
                         shift2, scale2, power2);
    if (!layer_scale2)
        LOG_FATAL("Failed to create scale layer");

    // C,H,W  C,1,1 -> C,H,W   (x - m) / (s + eps)^0.5)
    auto *layer_elwise2 = network.addElementWise(*layer_elwise1->getOutput(0),
                                                 *layer_scale2->getOutput(0),
                                                 ElementWiseOperation::kDIV);
    if (!layer_elwise2)
        LOG_FATAL("Failed to create elwise layer");

    auto *val3 =
        reinterpret_cast<float *>(malloc(sizeof(float) * scales.count));
    std::fill_n(val3, scales.count, 1.0f);
    Weights power3{DataType::kFLOAT, val3, scales.count};
    weightFactory.convert(power3);
    weightFactory.getTmpAllocs().push_back(val3);

    // C,H,W -> C,H,W    (x*scale + bias)^pow
    auto *scale3 = network.addScale(*layer_elwise2->getOutput(0),
                                    ScaleMode::kCHANNEL, bias, scales, power3);
    if (!scale3)
        LOG_FATAL("Failed to create scale layer");

    return scale3;
}
} // namespace nvcaffeparser1
