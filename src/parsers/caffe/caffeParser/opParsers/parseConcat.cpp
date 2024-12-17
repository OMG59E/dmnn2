/***
 * @Author: xingwg
 * @Date: 2024-12-09 15:27:39
 * @LastEditTime: 2024-12-13 12:50:25
 * @FilePath: /dmnn2/src/parsers/caffe/caffeParser/opParsers/parseConcat.cpp
 * @Description:
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#include "opParsers.h"

using namespace nvinfer1;

namespace nvcaffeparser1 {
ILayer *parseConcat(INetworkDefinition &network,
                    const trtcaffe::LayerParameter &msg,
                    CaffeWeightFactory & /*weightFactory*/,
                    BlobNameToTensor &tensors) {
    const trtcaffe::ConcatParameter &p = msg.concat_param();
    bool hasAxis = p.has_axis(); // optional parameter

    if (hasAxis && p.axis() < 0) {
        LOG_ERROR("Concat negative axis is not supported.");
        return nullptr;
    }
    if (network.hasImplicitBatchDimension() && p.axis() == 0) {
        LOG_ERROR("Concat across batch axis with implicit batch dimensions is "
                  "not supported.");
        return nullptr;
    }

    std::vector<ITensor *> ptrs;
    for (unsigned int i = 0, n = msg.bottom_size(); i < n; ++i)
        ptrs.push_back(tensors[msg.bottom().Get(i)]);

    auto concat = network.addConcatenation(&ptrs[0], msg.bottom_size());

    // If no axis is explicitly provided, do not call setAxis.
    // Rely on the default axis setting inside TRT which takes into account
    // NPCHW and higher dimensional input.
    if (hasAxis)
        concat->setAxis(p.axis() -
                        static_cast<int>(network.hasImplicitBatchDimension()));

    return concat;
}
} // namespace nvcaffeparser1
