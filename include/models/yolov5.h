/***
 * @Author: xingwg
 * @Date: 2024-12-16 11:04:46
 * @LastEditTime: 2024-12-16 11:04:55
 * @FilePath: /dmnn2/include/models/yolov5.h
 * @Description:
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#pragma once
#include "logging.h"
#include "net_operator.h"

namespace nv {
class YoloV5 : public NetOperator {
public:
    YoloV5() = default;
    ~YoloV5() = default;
    virtual int preprocess(const std::vector<nv::Image> &images);
    virtual int postprocess(const std::vector<nv::Image> &images,
                            std::vector<detections_t> &detections);

private:
    int min_wh_{2};
    int max_wh_{7680};
    float conf_threshold_{0.25f};
    float iou_threshold_{0.45f};
    float mean_vals_[3] = {0.0f, 0.0f, 0.0f};
    float std_vals_[3] = {255.0f, 255.0f, 255.0f};
    float padding_vals_[3] = {114.0f, 114.0f, 114.0f};
};
}  // namespace nv