/***
 * @Author: xingwg
 * @Date: 2024-10-18 10:12:04
 * @LastEditTime: 2024-10-21 11:58:29
 * @FilePath: /dmnn2/include/models/yolov6.h
 * @Description:
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#pragma once
#include "logging.h"
#include "net_operator.h"

namespace nv {
class YoloV6 : public NetOperator {
public:
    YoloV6() = default;
    ~YoloV6() = default;
    virtual int preprocess(const std::vector<nv::Image> &images);
    virtual int postprocess(const std::vector<nv::Image> &images,
                            std::vector<detections_t> &detections);

private:
    float conf_threshold_ = 0.25;
    float mean_vals_[3] = {0.0f, 0.0f, 0.0f};
    float std_vals_[3] = {255.0f, 255.0f, 255.0f};
    float padding_vals_[3] = {114.0f, 114.0f, 114.0f};
};
}  // namespace nv