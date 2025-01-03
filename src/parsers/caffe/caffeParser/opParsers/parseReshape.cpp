/***
 * @Author: xingwg
 * @Date: 2024-10-11 11:46:51
 * @LastEditTime: 2024-10-12 09:09:54
 * @FilePath: /dmnn2/src/parsers/caffe/caffeParser/opParsers/parseReshape.cpp
 * @Description:
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#include "opParsers.h"

using namespace nvinfer1;

namespace nvcaffeparser1 {
ILayer *parseReshape(INetworkDefinition &network,
                     const trtcaffe::LayerParameter &msg,
                     CaffeWeightFactory & /*weightFactory*/,
                     BlobNameToTensor &tensors) {
    if (!checkBlobs(msg, 1, 1))
        return nullptr;

    const trtcaffe::ReshapeParameter &p = msg.reshape_param();
    Dims bottomDims = tensors[msg.bottom(0)]->getDimensions();
    int axis = p.has_axis() ? p.axis() : 0;

    const ::trtcaffe::BlobShape &shape = p.shape();
    // Check that N (batch dim) is 0. TensorRT does not support reshape in batch
    // dimension
    if (network.hasImplicitBatchDimension() && (axis == 0) &&
        (shape.dim(0) != 0)) {
        LOG_ERROR("Invalid reshape param. TensorRT does not support reshape in "
                  "N (batch) dimension");
        return nullptr;
    }

    // Handle axis and dims parameters
    int axStart = std::max(0, axis - 1);
    int axEnd = p.has_num_axes()
                    ? std::max(0, axis -
                                      static_cast<int>(
                                          network.hasImplicitBatchDimension()) +
                                      p.num_axes())
                    : bottomDims.nbDims;

    std::vector<int> reshapeDims;
    reshapeDims.reserve(axStart);
    for (int i = 0; i < axStart; i++)
        reshapeDims.push_back(bottomDims.d[i]);

    for (int i = 0; i < shape.dim_size(); i++) {
        // skip first 0 (batch)
        if (network.hasImplicitBatchDimension() && axis == 0 && i == 0)
            continue;

        if (shape.dim(i) == 0) {
            // If there is no bottom dimension corresponding to the current
            // axis, then the params are invalid
            if (static_cast<int>(reshapeDims.size()) > bottomDims.nbDims)
                LOG_FATAL("Invalid reshape params");
            reshapeDims.push_back(bottomDims.d[reshapeDims.size()]);
        } else {
            reshapeDims.push_back(shape.dim(i));
        }
    }

    for (int i = axEnd; i < bottomDims.nbDims; i++)
        reshapeDims.push_back(bottomDims.d[i]);

    Dims topDims{};
    topDims.nbDims = static_cast<int>(reshapeDims.size());
    for (int i = 0; i < topDims.nbDims; i++)
        topDims.d[i] = reshapeDims[i];

    // Check there is at most one -1, and handle such case
    int countMinusOne = 0;
    for (int i = 0; i < topDims.nbDims; i++) {
        if (topDims.d[i] == -1) {
            countMinusOne += 1;
            // Inferred dimension
            int64_t newDim =
                parserutils::volume(bottomDims) / -parserutils::volume(topDims);
            topDims.d[i] = newDim;
        }
    }

    if (countMinusOne > 1) {
        LOG_ERROR("Invalid reshape param. At most one axis can be inferred "
                  "from other dimensions");
        return nullptr;
    }

    if (topDims.nbDims == 2) {
        topDims.nbDims += 1;
        topDims.d[2] = 1;
    }

    auto layer = network.addShuffle(*tensors[msg.bottom(0)]);
    layer->setReshapeDimensions(topDims);
    return layer;
}
} // namespace nvcaffeparser1
