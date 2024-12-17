/***
 * @Author: xingwg
 * @Date: 2024-10-11 11:46:51
 * @LastEditTime: 2024-12-13 12:13:17
 * @FilePath: /dmnn2/src/parsers/caffe/caffeParser/opParsers/parsePermute.cpp
 * @Description:
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#include "opParsers.h"

using namespace nvinfer1;

namespace nvcaffeparser1 {
ILayer *parsePermute(INetworkDefinition &network,
                     const trtcaffe::LayerParameter &msg,
                     CaffeWeightFactory & /*weightFactory*/,
                     BlobNameToTensor &tensors) {
    if (!checkBlobs(msg, 1, 1))
        return nullptr;

    const trtcaffe::PermuteParameter &p = msg.permute_param();
    Dims bottomDims = tensors[msg.bottom(0)]->getDimensions();
    Dims topDims = tensors[msg.bottom(0)]->getDimensions();
    int nbDims = bottomDims.nbDims;

    std::vector<int> orders;
    std::vector<bool> knownOrders(nbDims + 1, false);
    bool orderAbort =
        (p.order(0) != 0); // First order must be 0 (batch dimension)
    for (int i = 0; i < p.order_size(); i++) {
        int order = p.order(i);
        orderAbort |=
            (order > nbDims) ||
            (std::find(orders.begin(), orders.end(), order) != orders.end());
        orders.push_back(order);
        knownOrders[order] = true;
    }

    if (orderAbort) {
        LOG_ERROR(
            "Caffe Parser: Invalid permute param. TensorRT does not support "
            "permute in N (batch) dimension, and order index must be within "
            "the tensor dimensions. no duplicate order allowed.");
        return nullptr;
    }

    // Keep the rest of the order
    for (int i = 0; i < nbDims; i++) {
        if (!knownOrders[i])
            orders.push_back(i);
    }

    // Remove the first order (batch)
    orders.erase(orders.begin());

    for (int i = 0; i < nbDims; i++)
        topDims.d[i] = bottomDims.d[orders[i] - 1];

    if (parserutils::volume(topDims) != parserutils::volume(bottomDims))
        LOG_FATAL(
            "top dimensions volume does not match bottom dimensions volume");

    nvinfer1::Permutation permuteOrder;
    for (int i = 0; i < nbDims; i++)
        permuteOrder.order[i] = orders[i] - 1;

    auto permute = network.addShuffle(*tensors[msg.bottom(0)]);
    permute->setReshapeDimensions(topDims);
    permute->setFirstTranspose(permuteOrder);
    return permute;
}
} // namespace nvcaffeparser1