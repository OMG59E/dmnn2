/***
 * @Author: xingwg
 * @Date: 2024-12-09 15:27:39
 * @LastEditTime: 2024-12-13 12:13:44
 * @FilePath: /dmnn2/src/parsers/caffe/caffeParser/opParsers/parseInterp.cpp
 * @Description:
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#include "opParsers.h"

using namespace nvinfer1;

namespace nvcaffeparser1 {
ILayer *parseInterp(INetworkDefinition &network,
                    const trtcaffe::LayerParameter &msg,
                    CaffeWeightFactory & /*weightFactory*/,
                    BlobNameToTensor &tensors) {
    if (!checkBlobs(msg, 1, 1))
        return nullptr;

    const trtcaffe::InterpParameter &p = msg.interp_param();

    auto pad_beg = p.pad_beg();
    auto pad_end = p.pad_end();

    if (pad_beg > 0 || pad_end > 0) {
        LOG_ERROR("pad_beg && pad_end must be <= 0");
        return nullptr;
    }

    nv::DimsCHW dims =
        parserutils::getCHW(tensors[msg.bottom(0)]->getDimensions());
    // int in_c = dims.d[0];
    int in_h = dims.d[1];
    int in_w = dims.d[2];

    int height_in_eff = in_h + pad_beg + pad_end;
    int width_in_eff = in_w + pad_beg + pad_end;

    int out_h = 0;
    int out_w = 0;

    if (p.use_blob_size()) {
        LOG_ERROR("No support use_blob_size");
        return nullptr;
    } else if (p.has_shrink_factor() && !p.has_zoom_factor()) {
        const int shrink_factor = p.shrink_factor();
        if (shrink_factor < 1) {
            LOG_ERROR("shrink_factor must be >= 1");
            return nullptr;
        }
        out_h = (height_in_eff - 1) / shrink_factor + 1;
        out_w = (width_in_eff - 1) / shrink_factor + 1;
    } else if (p.has_zoom_factor() && !p.has_shrink_factor()) {
        const int zoom_factor = p.zoom_factor();
        if (zoom_factor < 1) {
            LOG_ERROR("zoom_factor must be >= 1");
            return nullptr;
        }
        out_h = height_in_eff + (height_in_eff - 1) * (zoom_factor - 1);
        out_w = width_in_eff + (width_in_eff - 1) * (zoom_factor - 1);
    } else if (p.has_height() && p.has_width()) {
        out_h = p.height();
        out_w = p.width();
    } else if (p.has_shrink_factor() && p.has_zoom_factor()) {
        const int shrink_factor = p.shrink_factor();
        const int zoom_factor = p.zoom_factor();

        if (shrink_factor < 1) {
            LOG_ERROR("shrink_factor must be >= 1");
            return nullptr;
        }
        if (zoom_factor < 1) {
            LOG_ERROR("zoom_factor must be >= 1");
            return nullptr;
        }

        out_h = (height_in_eff - 1) / shrink_factor + 1;
        out_w = (width_in_eff - 1) / shrink_factor + 1;
        out_h = out_h + (out_h - 1) * (zoom_factor - 1);
        out_w = out_w + (out_w - 1) * (zoom_factor - 1);
    }

    auto *layer = network.addResize(*tensors[msg.bottom(0)]);
    layer->setName(msg.name().c_str());
    layer->setResizeMode(ResizeMode::kLINEAR);
    layer->setOutputDimensions(Dims3{dims.d[0], out_h, out_w});
    return layer;
}
} // namespace nvcaffeparser1