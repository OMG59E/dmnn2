/***
 * @Author: xingwg
 * @Date: 2024-10-11 11:46:51
 * @LastEditTime: 2024-10-12 08:44:00
 * @FilePath: /dmnn2/src/parsers/caffe/caffeParser/caffeParser.cpp
 * @Description:
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#include "caffeParser.h"
#include "../../common/half.h"
#include "../../common/parserUtils.h"
#include "../binaryProtoBlob.h"
#include "base_types.h"
#include "error_check.h"
#include "google/protobuf/text_format.h"
#include "opParsers/opParsers.h"
#include "readProto.h"
#include <NvInferPluginUtils.h>
#include <iostream>

using namespace nvinfer1;
using namespace nvcaffeparser1;

CaffeParser::~CaffeParser() {
    for (auto v : mTmpAllocs) {
        free(v);
    }
    for (auto p : mNewPlugins) {
        if (p) {
            p->destroy();
        }
    }
    delete mBlobNameToTensor;
}

std::vector<nvinfer1::PluginField>
CaffeParser::parseNormalizeParam(const trtcaffe::LayerParameter &msg,
                                 CaffeWeightFactory &weightFactory,
                                 BlobNameToTensor &tensors) {
    std::vector<nvinfer1::PluginField> f;
    const trtcaffe::NormalizeParameter &p = msg.norm_param();

    int *acrossSpatial = allocMemory<int32_t>();
    *acrossSpatial = p.across_spatial() ? 1 : 0;
    f.emplace_back("acrossSpatial", acrossSpatial, PluginFieldType::kINT32, 1);

    int *channelShared = allocMemory<int32_t>();
    *channelShared = p.channel_shared() ? 1 : 0;
    f.emplace_back("channelShared", channelShared, PluginFieldType::kINT32, 1);

    auto *eps = allocMemory<float>();
    *eps = p.eps();
    f.emplace_back("eps", eps, PluginFieldType::kFLOAT32, 1);

    std::vector<Weights> w;
    // If .caffemodel is not provided, need to randomize the weight
    if (!weightFactory.isInitialized()) {
        int C =
            parserutils::getCHW(tensors[msg.bottom(0)]->getDimensions()).c();
        w.emplace_back(weightFactory.allocateWeights(
            C, std::normal_distribution<float>(0.0F, 1.0F)));
    } else {
        // Use the provided weight from .caffemodel
        w = weightFactory.getAllWeights(msg.name());
    }
    for (auto weight : w)
        f.emplace_back("weights", weight.values, PluginFieldType::kFLOAT32,
                       weight.count);
    int *nbWeights = allocMemory<int32_t>();
    *nbWeights = w.size();
    f.emplace_back("nbWeights", nbWeights, PluginFieldType::kINT32, 1);
    return f;
}

std::vector<nvinfer1::PluginField>
CaffeParser::parsePriorBoxParam(const trtcaffe::LayerParameter &msg,
                                CaffeWeightFactory &, BlobNameToTensor &) {
    std::vector<nvinfer1::PluginField> f;
    const trtcaffe::PriorBoxParameter &p = msg.prior_box_param();

    int minSizeSize = p.min_size_size();
    auto *minSize = allocMemory<float>(minSizeSize);
    for (int i = 0; i < minSizeSize; ++i)
        minSize[i] = p.min_size(i);
    f.emplace_back("minSize", minSize, PluginFieldType::kFLOAT32, minSizeSize);

    int maxSizeSize = p.max_size_size();
    auto *maxSize = allocMemory<float>(maxSizeSize);
    for (int i = 0; i < maxSizeSize; ++i)
        maxSize[i] = p.max_size(i);
    f.emplace_back("maxSize", maxSize, PluginFieldType::kFLOAT32, maxSizeSize);

    int aspectRatiosSize = p.aspect_ratio_size();
    auto *aspectRatios = allocMemory<float>(aspectRatiosSize);
    for (int i = 0; i < aspectRatiosSize; ++i)
        aspectRatios[i] = p.aspect_ratio(i);
    f.emplace_back("aspectRatios", aspectRatios, PluginFieldType::kFLOAT32,
                   aspectRatiosSize);

    int varianceSize = p.variance_size();
    auto *variance = allocMemory<float>(varianceSize);
    for (int i = 0; i < varianceSize; ++i)
        variance[i] = p.variance(i);
    f.emplace_back("variance", variance, PluginFieldType::kFLOAT32,
                   varianceSize);

    int *flip = allocMemory<int32_t>();
    *flip = p.flip() ? 1 : 0;
    f.emplace_back("flip", flip, PluginFieldType::kINT32, 1);

    int *clip = allocMemory<int32_t>();
    *clip = p.clip() ? 1 : 0;
    f.emplace_back("clip", clip, PluginFieldType::kINT32, 1);

    int *imgH = allocMemory<int32_t>();
    *imgH = p.has_img_h() ? p.img_h() : p.img_size();
    f.emplace_back("imgH", imgH, PluginFieldType::kINT32, 1);

    int *imgW = allocMemory<int32_t>();
    *imgW = p.has_img_w() ? p.img_w() : p.img_size();
    f.emplace_back("imgW", imgW, PluginFieldType::kINT32, 1);

    auto *stepH = allocMemory<float>();
    *stepH = p.has_step_h() ? p.step_h() : p.step();
    f.emplace_back("stepH", stepH, PluginFieldType::kFLOAT32, 1);

    auto *stepW = allocMemory<float>();
    *stepW = p.has_step_w() ? p.step_w() : p.step();
    f.emplace_back("stepW", stepW, PluginFieldType::kFLOAT32, 1);

    auto *offset = allocMemory<float>();
    *offset = p.offset();
    f.emplace_back("offset", offset, PluginFieldType::kFLOAT32, 1);
    return f;
}

std::vector<nvinfer1::PluginField>
CaffeParser::parseDetectionOutputParam(const trtcaffe::LayerParameter &msg,
                                       CaffeWeightFactory &,
                                       BlobNameToTensor &) {
    std::vector<nvinfer1::PluginField> f;
    const trtcaffe::DetectionOutputParameter &p = msg.detection_output_param();
    const trtcaffe::NonMaximumSuppressionParameter &nmsp = p.nms_param();

    int *shareLocation = allocMemory<int32_t>();
    *shareLocation = p.share_location() ? 1 : 0;
    f.emplace_back("shareLocation", shareLocation, PluginFieldType::kINT32, 1);

    int *varianceEncodedInTarget = allocMemory<int32_t>();
    *varianceEncodedInTarget = p.variance_encoded_in_target() ? 1 : 0;
    f.emplace_back("varianceEncodedInTarget", varianceEncodedInTarget,
                   PluginFieldType::kINT32, 1);

    int *backgroundLabelId = allocMemory<int32_t>();
    *backgroundLabelId = p.background_label_id();
    f.emplace_back("backgroundLabelId", backgroundLabelId,
                   PluginFieldType::kINT32, 1);

    int *numClasses = allocMemory<int32_t>();
    *numClasses = p.num_classes();
    f.emplace_back("numClasses", numClasses, PluginFieldType::kINT32, 1);

    // nms
    int *topK = allocMemory<int32_t>();
    *topK = nmsp.top_k();
    f.emplace_back("topK", topK, PluginFieldType::kINT32, 1);

    int *keepTopK = allocMemory<int32_t>();
    *keepTopK = p.keep_top_k();
    f.emplace_back("keepTopK", keepTopK, PluginFieldType::kINT32, 1);

    auto *confidenceThreshold = allocMemory<float>();
    *confidenceThreshold = p.confidence_threshold();
    f.emplace_back("confidenceThreshold", confidenceThreshold,
                   PluginFieldType::kFLOAT32, 1);

    // nms
    auto *nmsThreshold = allocMemory<float>();
    *nmsThreshold = nmsp.nms_threshold();
    f.emplace_back("nmsThreshold", nmsThreshold, PluginFieldType::kFLOAT32, 1);

    // input order = {0, 1, 2} in Caffe
    int *inputOrder = allocMemory<int32_t>(3);
    inputOrder[0] = 0;
    inputOrder[1] = 1;
    inputOrder[2] = 2;
    f.emplace_back("inputOrder", inputOrder, PluginFieldType::kINT32, 3);

    // confSigmoid = false for Caffe
    int *confSigmoid = allocMemory<int32_t>();
    *confSigmoid = 0;
    f.emplace_back("confSigmoid", confSigmoid, PluginFieldType::kINT32, 1);

    // isNormalized = true for Caffe
    int *isNormalized = allocMemory<int32_t>();
    *isNormalized = 1;
    f.emplace_back("isNormalized", isNormalized, PluginFieldType::kINT32, 1);

    // codeTypeSSD : from NvInferPlugin.h
    // CORNER = 0, CENTER_SIZE = 1, CORNER_SIZE = 2, TF_CENTER = 3
    int *codeType = allocMemory<int32_t>();
    switch (p.code_type()) {
    case trtcaffe::PriorBoxParameter::CORNER_SIZE:
        *codeType = static_cast<int>(plugin::CodeTypeSSD::CORNER_SIZE);
        break;
    case trtcaffe::PriorBoxParameter::CENTER_SIZE:
        *codeType = static_cast<int>(plugin::CodeTypeSSD::CENTER_SIZE);
        break;
    case trtcaffe::PriorBoxParameter::CORNER: // CORNER is default
    default:
        *codeType = static_cast<int>(plugin::CodeTypeSSD::CORNER);
        break;
    }
    f.emplace_back("codeType", codeType, PluginFieldType::kINT32, 1);

    int *numPts = allocMemory<int32_t>();
    *numPts = 1;
    f.emplace_back("numPts", numPts, PluginFieldType::kINT32, 1);

    return f;
}

std::vector<nvinfer1::PluginField>
CaffeParser::parseDetectionOutputV2Param(const trtcaffe::LayerParameter &msg,
                                         CaffeWeightFactory & /*weightFactory*/,
                                         BlobNameToTensor & /*tensors*/) {
    std::vector<nvinfer1::PluginField> f;
    const trtcaffe::DetectionOutputV2Parameter &p =
        msg.detection_output_v2_param();
    const trtcaffe::NonMaximumSuppressionParameter &nmsp = p.nms_param();

    int *shareLocation = allocMemory<int32_t>();
    *shareLocation = p.share_location() ? 1 : 0;
    f.emplace_back("shareLocation", shareLocation, PluginFieldType::kINT32, 1);

    int *varianceEncodedInTarget = allocMemory<int32_t>();
    *varianceEncodedInTarget = p.variance_encoded_in_target() ? 1 : 0;
    f.emplace_back("varianceEncodedInTarget", varianceEncodedInTarget,
                   PluginFieldType::kINT32, 1);

    int *backgroundLabelId = allocMemory<int32_t>();
    *backgroundLabelId = p.background_label_id();
    f.emplace_back("backgroundLabelId", backgroundLabelId,
                   PluginFieldType::kINT32, 1);

    int *numClasses = allocMemory<int32_t>();
    *numClasses = p.num_classes();
    f.emplace_back("numClasses", numClasses, PluginFieldType::kINT32, 1);

    // nms
    int *topK = allocMemory<int32_t>();
    *topK = nmsp.top_k();
    f.emplace_back("topK", topK, PluginFieldType::kINT32, 1);

    int *keepTopK = allocMemory<int32_t>();
    *keepTopK = p.keep_top_k();
    f.emplace_back("keepTopK", keepTopK, PluginFieldType::kINT32, 1);

    auto *confidenceThreshold = allocMemory<float>();
    *confidenceThreshold = p.confidence_threshold();
    f.emplace_back("confidenceThreshold", confidenceThreshold,
                   PluginFieldType::kFLOAT32, 1);

    // nms
    auto *nmsThreshold = allocMemory<float>();
    *nmsThreshold = nmsp.nms_threshold();
    f.emplace_back("nmsThreshold", nmsThreshold, PluginFieldType::kFLOAT32, 1);

    // input order = {0, 1, 2, 3} in Caffe
    int *inputOrder = allocMemory<int32_t>(4);
    inputOrder[0] = 0;
    inputOrder[1] = 1;
    inputOrder[2] = 2;
    inputOrder[3] = 3;
    f.emplace_back("inputOrder", inputOrder, PluginFieldType::kINT32, 4);

    // confSigmoid = false for Caffe
    int *confSigmoid = allocMemory<int32_t>();
    *confSigmoid = 0;
    f.emplace_back("confSigmoid", confSigmoid, PluginFieldType::kINT32, 1);

    // isNormalized = true for Caffe
    int *isNormalized = allocMemory<int32_t>();
    *isNormalized = 1;
    f.emplace_back("isNormalized", isNormalized, PluginFieldType::kINT32, 1);

    // codeTypeSSD : from NvInferPlugin.h
    // CORNER = 0, CENTER_SIZE = 1, CORNER_SIZE = 2, TF_CENTER = 3
    int *codeType = allocMemory<int32_t>();
    switch (p.code_type()) {
    case trtcaffe::PriorBoxParameter::CORNER_SIZE:
        *codeType = static_cast<int>(plugin::CodeTypeSSD::CORNER_SIZE);
        break;
    case trtcaffe::PriorBoxParameter::CENTER_SIZE:
        *codeType = static_cast<int>(plugin::CodeTypeSSD::CENTER_SIZE);
        break;
    case trtcaffe::PriorBoxParameter::CORNER: // CORNER is default
    default:
        *codeType = static_cast<int>(plugin::CodeTypeSSD::CORNER);
        break;
    }
    f.emplace_back("codeType", codeType, PluginFieldType::kINT32, 1);

    int *numPts = allocMemory<int32_t>();
    *numPts = p.num_pts();
    ;
    f.emplace_back("numPts", numPts, PluginFieldType::kINT32, 1);

    return f;
}

// std::vector<nvinfer1::PluginField> CaffeParser::parseLReLUParam(
//         const trtcaffe::LayerParameter &msg, CaffeWeightFactory &
//         /*weightFactory*/, BlobNameToTensor & /*tensors*/) {
//     std::vector<nvinfer1::PluginField> f;
//     const trtcaffe::ReLUParameter &p = msg.relu_param();
//
//     auto *negSlope = allocMemory<float>();
//     *negSlope = p.negative_slope();
//     f.emplace_back("negSlope", negSlope, PluginFieldType::kFLOAT32, 1);
//
//     return f;
// }

// std::vector<nvinfer1::PluginField> CaffeParser::parseFlattenParam(
//         const trtcaffe::LayerParameter &msg, CaffeWeightFactory &
//         /*weightFactory*/, BlobNameToTensor &tensors) {
//     std::vector<nvinfer1::PluginField> f;
//     const trtcaffe::FlattenParameter &p = msg.flatten_param();
//     auto *axis = allocMemory<int32_t>();
//     *axis = p.axis();
//     f.emplace_back("axis", axis, PluginFieldType::kINT32, 1);
//
//     nvinfer1::DimsCHW inputShape =
//     parserutils::getCHW(tensors[msg.bottom(0)]->getDimensions());
//
//     auto *endAxis = allocMemory<int32_t>();
//     *endAxis = p.end_axis();
//     f.emplace_back("end_axis", endAxis, PluginFieldType::kINT32, 1);
//
//     auto *iC = allocMemory<int32_t>();
//     *iC = inputShape.c();
//     f.emplace_back("channel_in", iC, PluginFieldType::kINT32, 1);
//
//     auto *iH = allocMemory<int32_t>();
//     *iH = inputShape.h();
//     f.emplace_back("height_in", iH, PluginFieldType::kINT32, 1);
//
//     auto *iW = allocMemory<int32_t>();
//     *iW = inputShape.w();
//     f.emplace_back("width_in", iW, PluginFieldType::kINT32, 1);
//
//     return f;
// }

// std::vector<nvinfer1::PluginField> CaffeParser::parseInterpParam(
//         const trtcaffe::LayerParameter &msg, CaffeWeightFactory &
//         /*weightFactory*/, BlobNameToTensor &tensors) {
//     std::vector<nvinfer1::PluginField> f;
//     const trtcaffe::InterpParameter &p = msg.interp_param();
//
//     auto *channels = allocMemory<int32_t>();
//     auto *height_in = allocMemory<int32_t>();
//     auto *width_in = allocMemory<int32_t>();
//     auto *height = allocMemory<int32_t>();
//     auto *width = allocMemory<int32_t>();
//     auto *pad_beg = allocMemory<int32_t>();
//     auto *pad_end = allocMemory<int32_t>();
//
//     *pad_beg = p.pad_beg();
//     *pad_end = p.pad_end();
//
//     LOG_ASSERT(*pad_beg <= 0 && *pad_end <= 0);
//
//     nvinfer1::DimsCHW inputShape =
//     parserutils::getCHW(tensors[msg.bottom(0)]->getDimensions());
//
//     *channels = inputShape.c();
//     *height_in = inputShape.h();
//     *width_in = inputShape.w();
//     int height_in_eff = *height_in + *pad_beg + *pad_end;
//     int width_in_eff = *width_in + *pad_beg + *pad_end;
//
//     if (p.use_blob_size()) {
//         LOG_ASSERT(msg.bottom().size() > 1);
//         *height =
//         parserutils::getCHW(tensors[msg.bottom(1)]->getDimensions()).h();
//         *width =
//         parserutils::getCHW(tensors[msg.bottom(1)]->getDimensions()).w();;
//     } else if (p.has_shrink_factor() && !p.has_zoom_factor()) {
//         const int shrink_factor = p.shrink_factor();
//         LOG_ASSERT(shrink_factor >= 1);
//         *height = (height_in_eff - 1) / shrink_factor + 1;
//         *width = (width_in_eff - 1) / shrink_factor + 1;
//     } else if (p.has_zoom_factor() && !p.has_shrink_factor()) {
//         const int zoom_factor = p.zoom_factor();
//         LOG_ASSERT(zoom_factor >= 1);
//         *height = height_in_eff + (height_in_eff - 1) * (zoom_factor - 1);
//         *width = width_in_eff + (width_in_eff - 1) * (zoom_factor - 1);
//     } else if (p.has_height() && p.has_width()) {
//         *height = p.height();
//         *width = p.width();
//     } else if (p.has_shrink_factor() && p.has_zoom_factor()) {
//         const int shrink_factor = p.shrink_factor();
//         const int zoom_factor = p.zoom_factor();
//
//         LOG_ASSERT(shrink_factor >= 1);
//         LOG_ASSERT(zoom_factor >= 1);
//
//         *height = (height_in_eff - 1) / shrink_factor + 1;
//         *width = (width_in_eff - 1) / shrink_factor + 1;
//
//         *height = *height + (*height - 1) * (zoom_factor - 1);
//         *width = *width + (*width - 1) * (zoom_factor - 1);
//     } else {
//         LOG_ASSERT(0);
//     }
//
//     f.emplace_back("channels", channels, PluginFieldType::kINT32, 1);
//     f.emplace_back("height_in", height_in, PluginFieldType::kINT32, 1);
//     f.emplace_back("width_in", width_in, PluginFieldType::kINT32, 1);
//     f.emplace_back("height_out", height, PluginFieldType::kINT32, 1);
//     f.emplace_back("width_out", width, PluginFieldType::kINT32, 1);
//     f.emplace_back("pad_end", pad_end, PluginFieldType::kINT32, 1);
//     f.emplace_back("pad_end", pad_end, PluginFieldType::kINT32, 1);
//
//     return f;
// }

std::vector<nvinfer1::PluginField>
CaffeParser::parseSliceParam(const trtcaffe::LayerParameter &msg,
                             CaffeWeightFactory & /*weightFactory*/,
                             BlobNameToTensor &tensors) {
    std::vector<nvinfer1::PluginField> f;
    const trtcaffe::SliceParameter &p = msg.slice_param();

    auto *axis = allocMemory<int32_t>(); // default = 1
    *axis = p.axis();
    f.emplace_back("axis", axis, PluginFieldType::kINT32, 1);

    auto *slice_dim = allocMemory<int32_t>(); // default = 1
    *slice_dim = p.slice_dim();
    f.emplace_back("slice_dim", slice_dim, PluginFieldType::kINT32, 1);

    int slice_point_size = p.slice_point_size();

    auto *slice_point = allocMemory<int32_t>(slice_point_size);
    for (int i = 0; i < slice_point_size; ++i)
        slice_point[i] = p.slice_point(i);
    f.emplace_back("slice_point", slice_point, PluginFieldType::kINT32,
                   slice_point_size);

    nv::DimsCHW inputShape =
        parserutils::getCHW(tensors[msg.bottom(0)]->getDimensions());

    auto *channel_in = allocMemory<int32_t>();
    *channel_in = inputShape.c();
    f.emplace_back("channel_in", channel_in, PluginFieldType::kINT32, 1);

    auto *height_in = allocMemory<int32_t>();
    *height_in = inputShape.h();
    f.emplace_back("height_in", height_in, PluginFieldType::kINT32, 1);

    auto *width_in = allocMemory<int32_t>();
    *width_in = inputShape.w();
    f.emplace_back("width_in", width_in, PluginFieldType::kINT32, 1);

    return f;
}

std::vector<nvinfer1::PluginField>
CaffeParser::parseScaleV2Param(const trtcaffe::LayerParameter &msg,
                               CaffeWeightFactory &weightFactory,
                               BlobNameToTensor &tensors) {
    std::vector<nvinfer1::PluginField> f;
    const trtcaffe::ScaleV2Parameter &p = msg.scale_v2_param();

    if (msg.bottom_size() != 2)
        LOG_FATAL("ScaleV2Layer must have 2 bottom blobs");

    int *axis = allocMemory<int32_t>();
    *axis = p.axis();
    f.emplace_back("axis", axis, PluginFieldType::kINT32, 1);

    int *num_axes = allocMemory<int32_t>();
    *num_axes = p.num_axes();
    f.emplace_back("num_axes", num_axes, PluginFieldType::kINT32, 1);

    if (p.bias_term())
        LOG_FATAL("ScaleV2Layer does not support bias_term");

    int *bias_term = allocMemory<int32_t>();
    *bias_term = p.bias_term() ? 1 : 0;
    f.emplace_back("bias_term", bias_term, PluginFieldType::kINT32, 1);

    return f;
}

// std::vector<nvinfer1::PluginField> CaffeParser::parseUpsampleParam(
//         const trtcaffe::LayerParameter &msg, CaffeWeightFactory
//         &weightFactory, BlobNameToTensor &tensors) {
//     std::vector<nvinfer1::PluginField> f;
//     const trtcaffe::UpsampleParameter &p = msg.upsample_param();
//     int *scale = allocMemory<int32_t>();
//     *scale = p.scale();
//     f.emplace_back("scale", scale, PluginFieldType::kINT32, 1);
//     return f;
// }

// std::vector<nvinfer1::PluginField> CaffeParser::parseMulParam(const
// trtcaffe::LayerParameter &msg,
//                                                               nvcaffeparser1::CaffeWeightFactory
//                                                               &weightFactory,
//                                                               nvcaffeparser1::BlobNameToTensor
//                                                               &tensors) {
//     std::vector<nvinfer1::PluginField> f;
//     const trtcaffe::MulParameter &p = msg.mul_param();
//     return f;
// }

// std::vector<nvinfer1::PluginField> CaffeParser::parseBroadcastMulParam(const
// trtcaffe::LayerParameter &msg,
//                                                                        nvcaffeparser1::CaffeWeightFactory
//                                                                        &weightFactory,
//                                                                        nvcaffeparser1::BlobNameToTensor
//                                                                        &tensors)
//                                                                        {
//     std::vector<nvinfer1::PluginField> f;
//     const trtcaffe::BroadcastMulParameter &p = msg.broadcat_mul_param();
//     return f;
// }

// std::vector<nvinfer1::PluginField> CaffeParser::parseAddParam(const
// trtcaffe::LayerParameter &msg,
//                                                               nvcaffeparser1::CaffeWeightFactory
//                                                               &weightFactory,
//                                                               nvcaffeparser1::BlobNameToTensor
//                                                               &tensors) {
//     std::vector<nvinfer1::PluginField> f;
//     const trtcaffe::AddParameter &p = msg.add_param();
//     auto *av = allocMemory<float>();
//     *av = p.av();
//     f.emplace_back("av", av, PluginFieldType::kFLOAT32, 1);
//
//     return f;
// }

// std::vector<nvinfer1::PluginField> CaffeParser::parseDivParam(const
// trtcaffe::LayerParameter &msg,
//                                                               nvcaffeparser1::CaffeWeightFactory
//                                                               &weightFactory,
//                                                               nvcaffeparser1::BlobNameToTensor
//                                                               &tensors) {
//     std::vector<nvinfer1::PluginField> f;
//     const trtcaffe::DivParameter &p = msg.div_param();
//     auto *dv = allocMemory<float>();
//     *dv = p.dv();
//     f.emplace_back("dv", dv, PluginFieldType::kFLOAT32, 1);
//
//     return f;
// }

// std::vector<nvinfer1::PluginField> CaffeParser::parseHardSwishParam(const
// trtcaffe::LayerParameter &msg,
//                                                                     nvcaffeparser1::CaffeWeightFactory
//                                                                     &weightFactory,
//                                                                     nvcaffeparser1::BlobNameToTensor
//                                                                     &tensors)
//                                                                     {
//     std::vector<nvinfer1::PluginField> f;
//     const trtcaffe::HardSwishParameter &p = msg.hardswish_param();
//     return f;
// }

std::vector<nvinfer1::PluginField>
CaffeParser::parseYoloBoxParam(const trtcaffe::LayerParameter &msg,
                               CaffeWeightFactory & /*weightFactory*/,
                               BlobNameToTensor & /*tensors*/) {
    std::vector<nvinfer1::PluginField> f;
    const auto &p = msg.yolo_box_param();
    int anchor_size = p.anchor_size();
    auto *anchor = allocMemory<float>(anchor_size);
    for (int i = 0; i < anchor_size; ++i)
        anchor[i] = p.anchor(i);
    f.emplace_back("anchor", anchor, PluginFieldType::kFLOAT32, anchor_size);

    auto *stride = allocMemory<float>();
    *stride = p.stride();
    f.emplace_back("stride", stride, PluginFieldType::kFLOAT32, 1);

    return f;
}

std::vector<nvinfer1::PluginField>
CaffeParser::parseYoloDetectionOutputParam(const trtcaffe::LayerParameter &msg,
                                           CaffeWeightFactory &weightFactory,
                                           BlobNameToTensor &tensors) {
    std::vector<nvinfer1::PluginField> f;
    const auto &p = msg.yolo_detection_output_param();

    auto *conf_thresh = allocMemory<float>();
    *conf_thresh = p.conf_thresh();
    f.emplace_back("conf_thresh", conf_thresh, PluginFieldType::kFLOAT32, 1);

    auto *nms_thresh = allocMemory<float>();
    *nms_thresh = p.nms_thresh();
    f.emplace_back("nms_thresh", nms_thresh, PluginFieldType::kFLOAT32, 1);

    int *keep_top_k = allocMemory<int32_t>();
    *keep_top_k = p.keep_top_k();
    f.emplace_back("keep_topK", keep_top_k, PluginFieldType::kINT32, 1);

    int *num_classes = allocMemory<int32_t>();
    *num_classes = p.num_classes();
    f.emplace_back("num_classes", num_classes, PluginFieldType::kINT32, 1);

    int *top_k = allocMemory<int32_t>();
    *top_k = p.top_k();
    f.emplace_back("topK", top_k, PluginFieldType::kINT32, 1);

    return f;
}

std::vector<nvinfer1::PluginField>
CaffeParser::parseFocusParam(const trtcaffe::LayerParameter &msg,
                             CaffeWeightFactory &weightFactory,
                             BlobNameToTensor &tensors) {
    std::vector<nvinfer1::PluginField> f;
    // const trtcaffe::FocusParameter &p = msg.focus_param();
    return f;
}

std::vector<nvinfer1::PluginField>
CaffeParser::parseYOLOXDetectionOutputParam(const trtcaffe::LayerParameter &msg,
                                            CaffeWeightFactory &weightFactory,
                                            BlobNameToTensor &tensors) {
    std::vector<nvinfer1::PluginField> f;
    const trtcaffe::YOLOXDetectionOutputParameter &p =
        msg.yolox_detection_output_param();

    auto *conf_thresh = allocMemory<float>();
    *conf_thresh = p.conf_thresh();
    f.emplace_back("conf_thresh", conf_thresh, PluginFieldType::kFLOAT32, 1);

    auto *nms_thresh = allocMemory<float>();
    *nms_thresh = p.nms_thresh();
    f.emplace_back("nms_thresh", nms_thresh, PluginFieldType::kFLOAT32, 1);

    int *keep_top_k = allocMemory<int32_t>();
    *keep_top_k = p.keep_top_k();
    f.emplace_back("keep_topK", keep_top_k, PluginFieldType::kINT32, 1);

    int *num_classes = allocMemory<int32_t>();
    *num_classes = p.num_classes();
    f.emplace_back("num_classes", num_classes, PluginFieldType::kINT32, 1);

    int *top_k = allocMemory<int32_t>();
    *top_k = p.top_k();
    f.emplace_back("topK", top_k, PluginFieldType::kINT32, 1);

    int *use_p6 = allocMemory<int32_t>();
    *use_p6 = p.use_p6();
    f.emplace_back("use_p6", use_p6, PluginFieldType::kINT32, 1);

    return f;
}

std::vector<nvinfer1::PluginField>
CaffeParser::parseCenterFaceOutputParam(const trtcaffe::LayerParameter &msg,
                                        CaffeWeightFactory &weightFactory,
                                        BlobNameToTensor &tensors) {
    std::vector<nvinfer1::PluginField> f;
    const trtcaffe::CenterFaceOutputParameter &p = msg.center_param();

    auto num_pts = allocMemory<int32_t>();
    *num_pts = p.num_pts();
    f.emplace_back("num_pts", num_pts, PluginFieldType::kINT32, 1);

    auto num_classes = allocMemory<int32_t>();
    *num_classes = p.num_classes();
    f.emplace_back("num_classes", num_classes, PluginFieldType::kINT32, 1);

    auto top_k = allocMemory<int32_t>();
    *top_k = p.top_k();
    f.emplace_back("topK", top_k, PluginFieldType::kINT32, 1);

    auto keep_top_k = allocMemory<int32_t>();
    *keep_top_k = p.keep_top_k();
    f.emplace_back("keep_topK", keep_top_k, PluginFieldType::kINT32, 1);

    auto nms_threshold = allocMemory<float>();
    *nms_threshold = p.nms_threshold();
    f.emplace_back("nms_threshold", nms_threshold, PluginFieldType::kFLOAT32,
                   1);

    auto confidence_threshold = allocMemory<float>();
    *confidence_threshold = p.confidence_threshold();
    f.emplace_back("confidence_threshold", confidence_threshold,
                   PluginFieldType::kFLOAT32, 1);

    return f;
}

const IBlobNameToTensor *
CaffeParser::parseBuffers(uint8_t const *deployBuffer, size_t deployLength,
                          uint8_t const *modelBuffer, size_t modelLength,
                          nvinfer1::INetworkDefinition &network,
                          nvinfer1::DataType weightType) noexcept {
    mDeploy =
        std::unique_ptr<trtcaffe::NetParameter>(new trtcaffe::NetParameter);
    google::protobuf::io::ArrayInputStream deployStream(deployBuffer,
                                                        deployLength);
    if (!google::protobuf::TextFormat::Parse(&deployStream, mDeploy.get())) {
        LOG_ERROR("Failed to parse deploy file");
        return nullptr;
    }
    if (modelBuffer) {
        mModel =
            std::unique_ptr<trtcaffe::NetParameter>(new trtcaffe::NetParameter);
        google::protobuf::io::ArrayInputStream modelStream(modelBuffer,
                                                           modelLength);
        google::protobuf::io::CodedInputStream codedModelStream(&modelStream);
        codedModelStream.SetTotalBytesLimit(modelLength, -1);
        if (!mModel->ParseFromCodedStream(&codedModelStream)) {
            LOG_ERROR("Failed to parse model file");
            return nullptr;
        }
    }
    return parse(network, weightType, modelBuffer != nullptr);
}

const IBlobNameToTensor *
CaffeParser::parse(char const *deployFile, char const *modelFile,
                   nvinfer1::INetworkDefinition &network,
                   nvinfer1::DataType weightType) noexcept {
    if (!deployFile) {
        LOG_ERROR("Deploy file is not specified");
        return nullptr;
    }

    // this is used to deal with dropout layers which have different input and
    // output
    mModel =
        std::unique_ptr<trtcaffe::NetParameter>(new trtcaffe::NetParameter);
    if (modelFile &&
        !readBinaryProto(mModel.get(), modelFile, mProtobufBufferSize)) {
        LOG_ERROR("Failed to parse model file");
        return nullptr;
    }

    mDeploy =
        std::unique_ptr<trtcaffe::NetParameter>(new trtcaffe::NetParameter);
    if (!readTextProto(mDeploy.get(), deployFile)) {
        LOG_ERROR("Failed to parse deploy file");
        return nullptr;
    }

    return parse(network, weightType, modelFile != nullptr);
}

const IBlobNameToTensor *CaffeParser::parse(INetworkDefinition &network,
                                            DataType weightType,
                                            bool hasModel) {
    bool ok = true;
    CaffeWeightFactory weights(*mModel.get(), weightType, mTmpAllocs, hasModel);
    mBlobNameToTensor = new (BlobNameToTensor);
    // Get list of all available plugin creators
    int numCreators = 0;
    nvinfer1::IPluginCreator *const *tmpList =
        getPluginRegistry()->getPluginCreatorList(&numCreators);
    for (int k = 0; k < numCreators; ++k) {
        if (!tmpList[k]) {
            LOG_WARNING("Plugin Creator for plugin {} is a nullptr", k);
            continue;
        }
        std::string pluginName = tmpList[k]->getPluginName();
        mPluginRegistry[pluginName] = tmpList[k];
    }

    for (int i = 0; i < mDeploy->input_size(); ++i) {
        Dims dims{0, 0, 0};
        if (network.hasImplicitBatchDimension()) {
            if (mDeploy->input_shape_size()) {
                dims = Dims3{(int)mDeploy->input_shape().Get(i).dim().Get(1),
                             (int)mDeploy->input_shape().Get(i).dim().Get(2),
                             (int)mDeploy->input_shape().Get(i).dim().Get(3)};
            } else {
                // Deprecated, but still used in a lot of networks
                dims = Dims3{(int)mDeploy->input_dim().Get(i * 4 + 1),
                             (int)mDeploy->input_dim().Get(i * 4 + 2),
                             (int)mDeploy->input_dim().Get(i * 4 + 3)};
            }
        } else {
            LOG_WARNING("Setting batch size to 1. Update the dimension after "
                        "parsing due to using explicit batch size.");
            if (mDeploy->input_shape_size()) {
                dims = Dims4{1, (int)mDeploy->input_shape().Get(i).dim().Get(1),
                             (int)mDeploy->input_shape().Get(i).dim().Get(2),
                             (int)mDeploy->input_shape().Get(i).dim().Get(3)};
            } else {
                // Deprecated, but still used in a lot of networks
                dims = Dims4{1, (int)mDeploy->input_dim().Get(i * 4 + 1),
                             (int)mDeploy->input_dim().Get(i * 4 + 2),
                             (int)mDeploy->input_dim().Get(i * 4 + 3)};
            }
        }
        ITensor *tensor = network.addInput(mDeploy->input().Get(i).c_str(),
                                           DataType::kFLOAT, dims);
        (*mBlobNameToTensor)[mDeploy->input().Get(i)] = tensor;
    }

    for (int i = 0; i < mDeploy->layer_size() && ok; ++i) {
        const trtcaffe::LayerParameter &layerMsg = mDeploy->layer(i);
        if (layerMsg.has_phase() && layerMsg.phase() == trtcaffe::TEST)
            continue;

        // If there is a inplace operation and the operation is
        // modifying the input, emit an error as
        for (int j = 0; ok && j < layerMsg.top_size(); ++j) {
            for (int k = 0; ok && k < layerMsg.bottom_size(); ++k) {
                if (layerMsg.top().Get(j) == layerMsg.bottom().Get(k)) {
                    auto iter =
                        mBlobNameToTensor->find(layerMsg.top().Get(j).c_str());
                    if (iter != nullptr && iter->isNetworkInput()) {
                        ok = false;
                        LOG_WARNING("TRT does not support in-place operations "
                                    "on input tensors in a prototxt file.");
                    }
                }
            }
        }

        if (getInferLibVersion() >= 5000) {
            if (mPluginFactoryV2 &&
                mPluginFactoryV2->isPluginV2(layerMsg.name().c_str())) {
                auto w = weights.getAllWeights(layerMsg.name());
                auto *plugin = mPluginFactoryV2->createPlugin(
                    layerMsg.name().c_str(), w.empty() ? nullptr : &w[0],
                    w.size(), mPluginNamespace.c_str());

                std::vector<ITensor *> inputs;
                for (int k = 0, n = layerMsg.bottom_size(); k < n; ++k)
                    inputs.push_back((*mBlobNameToTensor)[layerMsg.bottom(k)]);

                auto *layer = network.addPluginV2(&inputs[0],
                                                  int(inputs.size()), *plugin);
                if (!layer) {
                    LOG_ERROR("Failed to parsing layer type {}, index {}",
                              layerMsg.type(), i);
                    ok = false;
                }

                layer->setName(layerMsg.name().c_str());
                if (plugin->getNbOutputs() != layerMsg.top_size()) {
                    LOG_ERROR("Layer type: {} plugin: {} caffe: {}",
                              layerMsg.type(), plugin->getNbOutputs(),
                              layerMsg.top_size());
                    LOG_ERROR("Plugin layer output count is not equal to caffe "
                              "output count.");
                    ok = false;
                }

                for (int k = 0, n = std::min(layer->getNbOutputs(),
                                             layerMsg.top_size());
                     k < n; ++k)
                    (*mBlobNameToTensor)[layerMsg.top(k)] = layer->getOutput(k);

                continue;
            }

            // Use the TRT5 plugin creator method to check for built-in plugin
            // support
            std::string pluginName;
            nvinfer1::PluginFieldCollection fc{};
            std::vector<nvinfer1::PluginField> f;
            if (layerMsg.type() == "Normalize") {
                pluginName = "Normalize_TRT";
                f = parseNormalizeParam(layerMsg, weights, *mBlobNameToTensor);
            } else if (layerMsg.type() == "PriorBox") {
                pluginName = "PriorBox_TRT";
                f = parsePriorBoxParam(layerMsg, weights, *mBlobNameToTensor);
            } else if (layerMsg.type() == "DetectionOutput") {
                pluginName = "NMS_TRT";
                f = parseDetectionOutputParam(layerMsg, weights,
                                              *mBlobNameToTensor);
            } else if (layerMsg.type() == "DetectionOutputV2") {
                pluginName = "NMS_V2_TRT";
                f = parseDetectionOutputV2Param(layerMsg, weights,
                                                *mBlobNameToTensor);
            } else if (layerMsg.type() == "Slice") {
                pluginName = "Slice_TRT";
                f = parseSliceParam(layerMsg, weights, *mBlobNameToTensor);
            } else if (layerMsg.type() == "Focus") {
                pluginName = "Focus_TRT";
                f = parseFocusParam(layerMsg, weights, *mBlobNameToTensor);
            } else if (layerMsg.type() == "ScaleV2" ||
                       layerMsg.type() == "BroadcastMul") {
                pluginName = "ScaleV2_TRT";
                f = parseScaleV2Param(layerMsg, weights, *mBlobNameToTensor);
            } /*else if (layerMsg.type() == "Flatten") {
                pluginName = "Flatten_TRT";
                f = parseFlattenParam(layerMsg, weights, *mBlobNameToTensor);
            } else if (layerMsg.type() == "Interp") {
                pluginName = "Interp_TRT";
                f = parseInterpParam(layerMsg, weights, *mBlobNameToTensor);
            }*/
            // else if (layerMsg.type() == "Upsample")
            // {
            //     pluginName = "Upsample_TRT";
            //     f = parseUpsampleParam(layerMsg, weights,
            //     *mBlobNameToTensor);
            // } else if(layerMsg.type() == "HardSwish") {
            //     pluginName = "HardSwish_TRT";
            //     f = parseHardSwishParam(layerMsg, weights,
            //     *mBlobNameToTensor);
            // }
            // else if(layerMsg.type() == "Add") {
            //    pluginName = "Add_TRT";
            //    f = parseAddParam(layerMsg, weights, *mBlobNameToTensor);
            //} else if(layerMsg.type() == "Div") {
            //    pluginName = "Div_TRT";
            //    f = parseDivParam(layerMsg, weights, *mBlobNameToTensor);
            //} else if(layerMsg.type() == "BroadcastMul") {
            //    pluginName = "BroadcastMul_TRT";
            //    f = parseBroadcastMulParam(layerMsg, weights,
            //    *mBlobNameToTensor);
            //} else if(layerMsg.type() == "Mul") {
            //    pluginName = "Mul_TRT";
            //    f = parseMulParam(layerMsg, weights, *mBlobNameToTensor);
            //}
            else if (layerMsg.type() == "YoloBox") {
                pluginName = "YoloBox_TRT";
                f = parseYoloBoxParam(layerMsg, weights, *mBlobNameToTensor);
            } else if (layerMsg.type() == "YoloDetectionOutput") {
                pluginName = "YOLO_NMS_TRT";
                f = parseYoloDetectionOutputParam(layerMsg, weights,
                                                  *mBlobNameToTensor);
            } else if (layerMsg.type() == "CenterFaceOutput") {
                pluginName = "CT_NMS_TRT";
                f = parseCenterFaceOutputParam(layerMsg, weights,
                                               *mBlobNameToTensor);
            } else if (layerMsg.type() == "YOLOXDetectionOutput") {
                pluginName = "YOLOX_NMS_TRT";
                f = parseYOLOXDetectionOutputParam(layerMsg, weights,
                                                   *mBlobNameToTensor);
            }
            // else if(layerMsg.type() == "InstanceNorm") {
            //     pluginName = "InstanceNormalization_TRT";
            //     f = parseInstanceNormParam(layerMsg,weights,
            //     *mBlobNameToTensor);
            // }

            if (mPluginRegistry.find(pluginName) != mPluginRegistry.end()) {
                // Set fc
                fc.nbFields = static_cast<int32_t>(f.size());
                fc.fields = f.empty() ? nullptr : f.data();
                auto *pluginV2 =
                    mPluginRegistry.at(pluginName)
                        ->createPlugin(layerMsg.name().c_str(), &fc);
                if (!pluginV2)
                    LOG_FATAL("Plugin creation failed");
                mNewPlugins.push_back(pluginV2);

                std::vector<ITensor *> inputs;
                for (int k = 0, n = layerMsg.bottom_size(); k < n; ++k)
                    inputs.push_back((*mBlobNameToTensor)[layerMsg.bottom(k)]);

                auto layer = network.addPluginV2(&inputs[0], int(inputs.size()),
                                                 *pluginV2);
                if (!layer) {
                    LOG_ERROR("Failed to parsing layer type {} index {}",
                              layerMsg.type(), i);
                    ok = false;
                }
                layer->setName(layerMsg.name().c_str());
                if (pluginV2->getNbOutputs() != layerMsg.top_size()) {
                    LOG_ERROR("Layer type: {} plugin: {} caffe: {}",
                              layerMsg.type(), pluginV2->getNbOutputs(),
                              layerMsg.top_size());
                    LOG_ERROR("Plugin layer output count is not equal to caffe "
                              "output count.");
                    ok = false;
                }

                for (int k = 0, n = std::min(layer->getNbOutputs(),
                                             layerMsg.top_size());
                     k < n; ++k)
                    (*mBlobNameToTensor)[layerMsg.top(k)] = layer->getOutput(k);

                continue;
            }
        }

        if (layerMsg.type() == "Dropout") {
            (*mBlobNameToTensor)[layerMsg.top().Get(0)] =
                (*mBlobNameToTensor)[layerMsg.bottom().Get(0)];
            continue;
        }

        if (layerMsg.type() == "ContinuationIndicator") {
            (*mBlobNameToTensor)[layerMsg.top().Get(0)] =
                (*mBlobNameToTensor)[layerMsg.bottom().Get(0)];
            continue;
        }

        if (layerMsg.type() == "Input") {
            const trtcaffe::InputParameter &p = layerMsg.input_param();
            for (int k = 0; k < layerMsg.top_size(); ++k) {
                const trtcaffe::BlobShape &shape = p.shape().Get(k);
                if (shape.dim_size() != 4) {
                    LOG_ERROR("error parsing input layer, TensorRT only "
                              "supports 4 dimensional input");
                    return nullptr;
                } else {
                    Dims d;
                    if (network.hasImplicitBatchDimension()) {
                        d = Dims3{(int)shape.dim().Get(1),
                                  (int)shape.dim().Get(2),
                                  (int)shape.dim().Get(3)};
                    } else {
                        d = Dims4{
                            (int)shape.dim().Get(0), (int)shape.dim().Get(1),
                            (int)shape.dim().Get(2), (int)shape.dim().Get(3)};
                    }
                    auto *tensor = network.addInput(layerMsg.top(k).c_str(),
                                                    DataType::kFLOAT, d);
                    (*mBlobNameToTensor)[layerMsg.top().Get(k)] = tensor;
                }
            }
            continue;
        }

        // Use parser table to lookup the corresponding parse function to handle
        // the rest of the layers
        auto v = gParseTable.find(layerMsg.type());

        if (v == gParseTable.end()) {
            LOG_ERROR("Could not parse layer type {}", layerMsg.type());
            ok = false;
        } else {
            auto *layer =
                (*v->second)(network, layerMsg, weights, *mBlobNameToTensor);
            if (!layer) {
                LOG_ERROR("error parse layer type {} index {}", layerMsg.type(),
                          i);
                ok = false;
            } else {
                layer->setName(layerMsg.name().c_str());
                (*mBlobNameToTensor)[layerMsg.top(0)] = layer->getOutput(0);
            }
        }
    }

    mBlobNameToTensor->setTensorNames();

    return ok && weights.isOK() && mBlobNameToTensor->isOK() ? mBlobNameToTensor
                                                             : nullptr;
}

IBinaryProtoBlob *CaffeParser::parseBinaryProto(const char *fileName) noexcept {
    if (!fileName) {
        LOG_ERROR("input filename is null");
        return nullptr;
    }

    using namespace google::protobuf::io;

    std::ifstream stream(fileName, std::ios::in | std::ios::binary);
    if (!stream) {
        LOG_ERROR("Could not open file {}", fileName);
        return nullptr;
    }

    IstreamInputStream rawInput(&stream);
    CodedInputStream codedInput(&rawInput);
    codedInput.SetTotalBytesLimit(INT_MAX, -1);

    trtcaffe::BlobProto blob;
    bool ok = blob.ParseFromCodedStream(&codedInput);
    stream.close();

    if (!ok) {
        LOG_ERROR("Could not parse mean file");
        return nullptr;
    }

    Dims4 dims{1, 1, 1, 1};
    if (blob.has_shape()) {
        int size = blob.shape().dim_size(), s[4] = {1, 1, 1, 1};
        for (int i = 4 - size; i < 4; i++) {
            if (blob.shape().dim(i) >= INT32_MAX)
                LOG_FATAL("Invalid dimensions {}, must be < {}",
                          blob.shape().dim(i), INT32_MAX);
            s[i] = static_cast<int>(blob.shape().dim(i));
        }
        dims = Dims4{s[0], s[1], s[2], s[3]};
    } else {
        dims = Dims4{blob.num(), blob.channels(), blob.height(), blob.width()};
    }

    const int dataSize = dims.d[0] * dims.d[1] * dims.d[2] * dims.d[3];
    if (dataSize < 0)
        LOG_FATAL("Invalid shape size {}, must be > 0", dataSize);

    const trtcaffe::Type blobProtoDataType =
        CaffeWeightFactory::getBlobProtoDataType(blob);
    const auto blobProtoData = CaffeWeightFactory::getBlobProtoData(
        blob, blobProtoDataType, mTmpAllocs);

    if (dataSize != (int)blobProtoData.second) {
        LOG_ERROR("blob dimensions don't match data size.");
        return nullptr;
    }

    const auto dataSizeBytes =
        dataSize * CaffeWeightFactory::sizeOfCaffeType(blobProtoDataType);
    void *memory = malloc(dataSizeBytes);
    memcpy(memory, blobProtoData.first, dataSizeBytes);
    return new BinaryProtoBlob(memory,
                               blobProtoDataType == trtcaffe::FLOAT
                                   ? DataType::kFLOAT
                                   : DataType::kHALF,
                               dims);
}
