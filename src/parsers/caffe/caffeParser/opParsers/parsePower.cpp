/***
 * @Author: xingwg
 * @Date: 2024-10-11 11:46:51
 * @LastEditTime: 2024-10-12 08:59:58
 * @FilePath: /dmnn2/src/parsers/caffe/caffeParser/opParsers/parsePower.cpp
 * @Description:
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#include "opParsers.h"

using namespace nvinfer1;

namespace nvcaffeparser1 {
ILayer *parsePower(INetworkDefinition &network,
                   const trtcaffe::LayerParameter &msg,
                   CaffeWeightFactory &weightFactory,
                   BlobNameToTensor &tensors) {
    if (!checkBlobs(msg, 1, 1))
        return nullptr;

    const trtcaffe::PowerParameter &p = msg.power_param();

    float shift = p.has_shift() ? p.shift() : 0.0f;
    float scale = p.has_scale() ? p.scale() : 1.0f;
    float power = p.has_power() ? p.power() : 1.0f;

    DataType dataType = weightFactory.getDataType();
    assert(dataType == DataType::kFLOAT || dataType == DataType::kHALF);

    Weights wShift, wScale, wPower;
    if (dataType == DataType::kHALF) {
        auto *t = reinterpret_cast<float16 *>(malloc(3 * sizeof(float16)));
        t[0] = float16(shift), t[1] = float16(scale), t[2] = float16(power);
        wShift = Weights{DataType::kHALF, &t[0], 1};
        wScale = Weights{DataType::kHALF, &t[1], 1};
        wPower = Weights{DataType::kHALF, &t[2], 1};
        weightFactory.getTmpAllocs().push_back(t);
    } else {
        auto *t = reinterpret_cast<float *>(malloc(3 * sizeof(float)));
        t[0] = shift, t[1] = scale, t[2] = power;
        wShift = Weights{DataType::kFLOAT, &t[0], 1};
        wScale = Weights{DataType::kFLOAT, &t[1], 1};
        wPower = Weights{DataType::kFLOAT, &t[2], 1};
        weightFactory.getTmpAllocs().push_back(t);
    }

    weightFactory.convert(wShift);
    weightFactory.convert(wScale);
    weightFactory.convert(wPower);
    return network.addScale(*tensors[msg.bottom(0)], ScaleMode::kUNIFORM,
                            wShift, wScale, wPower);
}
} // namespace nvcaffeparser1