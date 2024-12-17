/*
 * @Author: xingwg
 * @Date: 2024-12-16 11:04:30
 * @LastEditTime: 2024-12-16 16:46:30
 * @FilePath: /dmnn2/src/models/yolov5.cc
 * @Description:
 *
 * Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#include "models/yolov5.h"
#include "imgproc/resize.h"
#include "utils/nms.h"

namespace nv {
int YoloV5::preprocess(const std::vector<nv::Image> &images) {
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

int YoloV5::postprocess(const std::vector<nv::Image> &images,
                        std::vector<detections_t> &detections) {
    detections.clear();
    LOG_ASSERT(getNbOutputs() == 1 && getNbInputs() == 1);
    auto &input = inputs_[0];
    auto &output = outputs_[0];
    LOG_ASSERT(input.nbDims == 4 && output.nbDims == 3);
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
        int num_anchors = output.dims[1];
        int step = output.dims[2];
        int num_classes = step - 5;
        LOG_INFO("num_anchors={} step={} num_classes={}", num_anchors, step,
                 num_classes);
        float *pred = (float *)(output.data) + n * num_anchors * step;
        for (int k = 0; k < num_anchors; ++k) {
            float obj_conf = pred[k * step + 4];
            if (obj_conf < conf_threshold_)
                continue;

            float w = pred[k * step + 2];
            float h = pred[k * step + 3];

            if (w < min_wh_ || h < min_wh_ || w > max_wh_ || h > max_wh_)
                continue;

            float cx = pred[k * step + 0];
            float cy = pred[k * step + 1];

            // scale_coords
            int x1 = int((cx - w * 0.5f - start_x) * scale);
            int y1 = int((cy - h * 0.5f - start_y) * scale);
            int x2 = int((cx + w * 0.5f - start_x) * scale);
            int y2 = int((cy + h * 0.5f - start_y) * scale);

            // clip
            x1 = x1 < 0 ? 0 : x1;
            y1 = y1 < 0 ? 0 : y1;
            x2 = x2 >= images[n].w() ? images[n].w() - 1 : x2;
            y2 = y2 >= images[n].h() ? images[n].h() - 1 : y2;

            detection_t detection;
            detection.bbox.x1 = x1;
            detection.bbox.y1 = y1;
            detection.bbox.x2 = x2;
            detection.bbox.y2 = y2;
            int num_cls{-1};
            float max_conf{-1};
            for (int dc = 0; dc < num_classes; ++dc) {  // [0-80)
                float conf = pred[k * step + 5 + dc] * obj_conf;
                if (max_conf < conf) {
                    num_cls = dc;
                    max_conf = conf;
                }
            }
            if (max_conf < conf_threshold_)
                continue;
            detection.cls_idx = num_cls;
            detection.score = max_conf;
            image_detections.emplace_back(detection);
        }
        if (!image_detections.empty()) {
            // nms
            non_max_suppression(image_detections, iou_threshold_);
        }
        detections.emplace_back(image_detections);
    }
    return 0;
}
}  // namespace nv