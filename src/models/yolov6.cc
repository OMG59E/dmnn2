/*
 * @Author: xingwg
 * @Date: 2024-10-18 10:31:22
 * @LastEditTime: 2024-12-16 11:34:12
 * @FilePath: /dmnn2/src/models/yolov6.cc
 * @Description:
 *
 * Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#include "models/yolov6.h"
#include "imgproc/resize.h"

namespace nv {
int YoloV6::preprocess(const std::vector<nv::Image> &images) {
    LOG_ASSERT(getNbInputs() == 1);
    auto &input = inputs_[0];
    LOG_ASSERT(input.nbDims == 4);
    uint64_t offset = 0;
    for (auto &image : images) {
        nv::Image dst;
        dst.own = false;
        dst.gpu_data = input.gpu_data + offset;
        dst.height = input.dims[2];
        dst.width = input.dims[3];
        dst.dataType = input.dataType;
        dst.colorType = nv::COLOR_TYPE_RGB888_PLANAR;
        // resize -> padding -> cvtColor -> norm
        if (0 != resizePaddingCvtColorNormAsync(stream_, image, dst, mean_vals_,
                                                std_vals_, PaddingMode::CENTER,
                                                padding_vals_)) {
            LOG_ERROR("resizePaddingCvtColorNormAsync failed");
            return -1;
        }
        offset += input.size_bytes();
    }
    valid_batch_size_ = images.size();
    return 0;
}

int YoloV6::postprocess(const std::vector<nv::Image> &images,
                        std::vector<detections_t> &detections) {
    detections.clear();
    LOG_ASSERT(getNbOutputs() == 1 && getNbInputs() == 1);
    auto &input = inputs_[0];
    auto &output = outputs_[0];
    LOG_ASSERT(input.nbDims == 4 && output.nbDims == 4);
    float input_h = input.dims[2];
    float input_w = input.dims[3];
    // batch
    for (int n = 0; n < images.size(); ++n) {
        auto &image = images[n];
        float sx = image.w() / (float)input_w;
        float sy = image.h() / (float)input_h;
        float scale = 0;
        float ratio = image.w() / (float)image.h();
        float start_x = 0;
        float start_y = 0;
        if (sx > sy) {
            start_y = (input_h - input_w / ratio) * 0.5f;
            scale = sx;
        } else {
            start_x = (input_w - input_h * ratio) * 0.5f;
            scale = sy;
        }
        detections_t image_detections;
        image_detections.clear();
        int num_proprosals = output.dims[2];
        int stride = output.dims[3];
        float *detection_out =
            (float *)(output.data) + n * num_proprosals * stride;
        for (int k = 0; k < num_proprosals; ++k) {
            if (*(detection_out + k * stride + 2) < conf_threshold_)
                break;
            detection_t detection;
            detection.cls_idx = int(*(detection_out + k * stride + 1));
            detection.score = *(detection_out + k * stride + 2);
            detection.bbox.x1 =
                int((*(detection_out + k * stride + 3) - start_x) * scale);
            detection.bbox.y1 =
                int((*(detection_out + k * stride + 4) - start_y) * scale);
            detection.bbox.x2 =
                int((*(detection_out + k * stride + 5) - start_x) * scale);
            detection.bbox.y2 =
                int((*(detection_out + k * stride + 6) - start_y) * scale);
            // clip
            detection.bbox.x1 = std::max(0, detection.bbox.x1);
            detection.bbox.y1 = std::max(0, detection.bbox.y1);
            detection.bbox.x2 = std::min(image.w() - 1, detection.bbox.x2);
            detection.bbox.y2 = std::min(image.h() - 1, detection.bbox.y2);
            image_detections.emplace_back(detection);
        }
        detections.emplace_back(image_detections);
    }
    return 0;
}
}  // namespace nv