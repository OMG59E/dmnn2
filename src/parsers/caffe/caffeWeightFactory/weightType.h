/***
 * @Author: xingwg
 * @Date: 2024-10-11 11:46:51
 * @LastEditTime: 2024-10-12 09:20:29
 * @FilePath: /dmnn2/src/parsers/caffe/caffeWeightFactory/weightType.h
 * @Description:
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#ifndef TRT_CAFFE_PARSER_WEIGHT_TYPE_H
#define TRT_CAFFE_PARSER_WEIGHT_TYPE_H

namespace nvcaffeparser1 {
enum class WeightType {
    // types for convolution, deconv, fully connected
    kGENERIC = 0, // typical weights for the layer: e.g. filter (for conv) or
                  // matrix weights (for innerproduct)
    kBIAS = 1,    // bias weights

    // These enums are for BVLCCaffe, which are incompatible with nvCaffe enums
    // below.
    // See batch_norm_layer.cpp in BLVC source of Caffe
    kMEAN = 0,
    kVARIANCE = 1,
    kMOVING_AVERAGE = 2,

    // These enums are for nvCaffe, which are incompatible with BVLCCaffe enums
    // above
    // See batch_norm_layer.cpp in NVidia fork of Caffe
    kNVMEAN = 0,
    kNVVARIANCE = 1,
    kNVSCALE = 3,
    kNVBIAS = 4
};
} // namespace nvcaffeparser1
#endif // TRT_CAFFE_PARSER_WEIGHT_TYPE_H