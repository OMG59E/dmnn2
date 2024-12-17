/***
 * @Author: xingwg
 * @Date: 2024-10-11 11:46:51
 * @LastEditTime: 2024-10-12 09:10:15
 * @FilePath: /dmnn2/src/parsers/caffe/caffeParser/opParsers/parseSoftMax.cpp
 * @Description:
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#include "opParsers.h"

using namespace nvinfer1;

namespace nvcaffeparser1 {
ILayer *parseSoftMax(INetworkDefinition &network,
                     const trtcaffe::LayerParameter &msg,
                     CaffeWeightFactory & /*weightFactory*/,
                     BlobNameToTensor &tensors) {
    if (!checkBlobs(msg, 1, 1))
        return nullptr;

    const trtcaffe::SoftmaxParameter &p = msg.softmax_param();

    // Caffe supports negative axis, indexing from the last dimension
    // However, there is a discrepancy in the internal tensor dimension in some
    // cases. For example. InnerProduct produces flat 1D blob in Caffe, while
    // TensorRT still produces CHW format. MNIST sample generates input to
    // Softmax as,
    //     Caffe    = n x 10
    //     TensorRT = n x 10 x 1 x 1
    // To make sure we do not run into issues, negative axis won't be supported
    // in TensorRT
    int nbDims = tensors[msg.bottom(0)]->getDimensions().nbDims;
    bool hasAxis = p.has_axis();       // optional parameter
    int axis = hasAxis ? p.axis() : 1; // default is 1

    if (network.hasImplicitBatchDimension() && axis == 0) {
        LOG_ERROR(
            "Invalid axis in softmax layer - TensorRT does not support softmax "
            "across the batch axis with implicit batch dimensions networks.");
        return nullptr;
    }

    if (axis < 0 || axis > 3 || (axis > nbDims)) {
        LOG_ERROR("Invalid axis in softmax layer - TensorRT expects NCHW "
                  "input. Negative axis is not supported in TensorRT, please "
                  "use positive axis indexing");
        return nullptr;
    }

    auto softmax = network.addSoftMax(*tensors[msg.bottom(0)]);
    // Do this so that setAxes is not used when the default axis is needed
    // This is necessary to preserve correct roll-into-the-batch dimension
    // behaviour for samples like FasterRCNN NCHW -> default axis when setAxes
    // is not called will be 1 (the C dimension) NPCHW -> default axis when
    // setAxes is not called will be 2 (the C dimension)
    if (hasAxis) {
        uint32_t axes = 1u << (axis - static_cast<int>(
                                          network.hasImplicitBatchDimension()));
        softmax->setAxes(axes);
    }
    return softmax;
}
} // namespace nvcaffeparser1
