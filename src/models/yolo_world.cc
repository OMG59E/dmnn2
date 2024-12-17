/*
 * @Author: xingwg
 * @Date: 2024-12-11 17:42:43
 * @LastEditTime: 2024-12-16 17:21:40
 * @FilePath: /dmnn2/src/models/yolo_world.cc
 * @Description:
 *
 * Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#include "models/yolo_world.h"
#include "imgproc/resize.h"

namespace nv {
int YOLOWorld::preprocess(const std::vector<nv::Image> &images) {
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

int YOLOWorld::postprocess(const std::vector<nv::Image> &images,
                           std::vector<detections_t> &detections) {
    detections.clear();
    LOG_ASSERT(getNbOutputs() == 4 && getNbInputs() == 1);
    auto &input = inputs_[0];
    auto &num_dets = outputs_[0];  // 1x1
    auto &boxes = outputs_[1];     // 1x100x4
    auto &scores = outputs_[2];    // 1x100
    auto &labels = outputs_[3];    // 1x100
    LOG_ASSERT(num_dets.dataType == nv::DataType::DATA_TYPE_INT32);
    LOG_ASSERT(boxes.dataType == nv::DataType::DATA_TYPE_FLOAT32);
    LOG_ASSERT(scores.dataType == nv::DataType::DATA_TYPE_FLOAT32);
    LOG_ASSERT(labels.dataType == nv::DataType::DATA_TYPE_INT32);

    int *num_dets_data = (int *)num_dets.data;
    float *boxes_data = (float *)boxes.data;
    float *scores_data = (float *)scores.data;
    int *labels_data = (int *)labels.data;

    float input_h = input.dims[2];
    float input_w = input.dims[3];
    int num_bboxes = boxes.dims[1];
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
        LOG_INFO("batch{}: num_dets={}", n, num_dets_data[n]);
        for (int k = 0; k < num_dets_data[n]; ++k) {
            detection_t detection;
            detection.cls_idx = labels_data[n * num_bboxes + k];
            detection.score = scores_data[n * num_bboxes + k];
            if (detection.score < conf_threshold_)
                continue;
            detection.bbox.x1 = int(
                (boxes_data[n * num_bboxes * 4 + k * 4 + 0] - start_x) * scale);
            detection.bbox.y1 = int(
                (boxes_data[n * num_bboxes * 4 + k * 4 + 1] - start_y) * scale);
            detection.bbox.x2 = int(
                (boxes_data[n * num_bboxes * 4 + k * 4 + 2] - start_x) * scale);
            detection.bbox.y2 = int(
                (boxes_data[n * num_bboxes * 4 + k * 4 + 3] - start_y) * scale);
            if (detection.bbox.x1 < 0)
                detection.bbox.x1 = 0;
            if (detection.bbox.y1 < 0)
                detection.bbox.y1 = 0;
            if (detection.bbox.x2 >= image.w())
                detection.bbox.x2 = image.w() - 1;
            if (detection.bbox.y2 >= image.h())
                detection.bbox.y2 = image.h() - 1;
            image_detections.emplace_back(detection);
            LOG_INFO("batch{}, detection output: {:2d} label={:4d} "
                     "score={:.6f} x1={:4d} "
                     "y1={:4d} x2={:4d} y2={:4d}",
                     n, k, detection.cls_idx, detection.score,
                     detection.bbox.x1, detection.bbox.y1, detection.bbox.x2,
                     detection.bbox.y2);
        }
        detections.emplace_back(image_detections);
    }
    return 0;
}
}  // namespace nv