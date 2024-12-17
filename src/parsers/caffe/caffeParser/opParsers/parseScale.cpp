/***
 * @Author: xingwg
 * @Date: 2024-12-09 15:27:39
 * @LastEditTime: 2024-12-13 12:11:46
 * @FilePath: /dmnn2/src/parsers/caffe/caffeParser/opParsers/parseScale.cpp
 * @Description:
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
/***
 * @Author: xingwg
 * @Date: 2024-10-11 11:46:51
 * @LastEditTime: 2024-10-12 09:08:47
 * @FilePath: /dmnn2/src/parsers/caffe/caffeParser/opParsers/parseScale.cpp
 * @Description:
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#include "opParsers.h"

using namespace nvinfer1;

namespace nvcaffeparser1 {
ILayer *parseScale(INetworkDefinition &network,
                   const trtcaffe::LayerParameter &msg,
                   CaffeWeightFactory &weightFactory,
                   BlobNameToTensor &tensors) {
    if (!checkBlobs(msg, 1, 1))
        return nullptr;

    const trtcaffe::ScaleParameter &p = msg.scale_param();
    int C = parserutils::getCHW(tensors[msg.bottom(0)]->getDimensions()).c();

    Weights scale =
        weightFactory.isInitialized()
            ? weightFactory(msg.name(), WeightType::kGENERIC)
            : weightFactory.allocateWeights(
                  C, std::uniform_real_distribution<float>(0.9F, 1.1F));
    Weights shift = !p.has_bias_term() || p.bias_term()
                        ? (weightFactory.isInitialized()
                               ? weightFactory(msg.name(), WeightType::kBIAS)
                               : weightFactory.allocateWeights(C))
                        : weightFactory.getNullWeights();
    Weights power = weightFactory.getNullWeights();

    weightFactory.convert(shift);
    weightFactory.convert(scale);
    weightFactory.convert(power);

    return network.addScale(*tensors[msg.bottom(0)], ScaleMode::kCHANNEL, shift,
                            scale, power);
}
} // namespace nvcaffeparser1