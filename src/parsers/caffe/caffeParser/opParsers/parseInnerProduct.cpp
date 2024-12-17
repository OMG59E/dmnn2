/***
 * @Author: xingwg
 * @Date: 2024-10-11 11:46:51
 * @LastEditTime: 2024-10-12 08:50:57
 * @FilePath:
 * /dmnn2/src/parsers/caffe/caffeParser/opParsers/parseInnerProduct.cpp
 * @Description:
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#include "opParsers.h"

using namespace nvinfer1;

namespace nvcaffeparser1 {
ILayer *parseInnerProduct(INetworkDefinition &network,
                          const trtcaffe::LayerParameter &msg,
                          CaffeWeightFactory &weightFactory,
                          BlobNameToTensor &tensors) {
    const trtcaffe::InnerProductParameter &p = msg.inner_product_param();

    int64_t nbInputs =
        parserutils::getCHW(tensors[msg.bottom(0)]->getDimensions()).size();
    int64_t nbOutputs = p.num_output();

    float std_dev = 1.0f / sqrtf(nbInputs * nbOutputs);
    Weights kernelWeights =
        weightFactory.isInitialized()
            ? weightFactory(msg.name(), WeightType::kGENERIC)
            : weightFactory.allocateWeights(
                  nbInputs * nbOutputs,
                  std::normal_distribution<float>(0.0f, std_dev));
    Weights biasWeights =
        !p.has_bias_term() || p.bias_term()
            ? (weightFactory.isInitialized()
                   ? weightFactory(msg.name(), WeightType::kBIAS)
                   : weightFactory.allocateWeights(nbOutputs))
            : weightFactory.getNullWeights();

    weightFactory.convert(kernelWeights);
    weightFactory.convert(biasWeights);
    return network.addFullyConnected(*tensors[msg.bottom(0)], p.num_output(),
                                     kernelWeights, biasWeights);
}
} // namespace nvcaffeparser1