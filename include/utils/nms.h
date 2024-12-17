/***
 * @Author: xingwg
 * @Date: 2024-12-16 11:28:04
 * @LastEditTime: 2024-12-16 11:28:17
 * @FilePath: /dmnn2/include/utils/nms.h
 * @Description:
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#pragma once
#include "base_types.h"
#include <algorithm>
#include <vector>

inline float bbox_overlap(const nv::BoundingBox &vi,
                          const nv::BoundingBox &vo) {
    int xx1 = std::max(vi.x1, vo.x1);
    int yy1 = std::max(vi.y1, vo.y1);
    int xx2 = std::min(vi.x2, vo.x2);
    int yy2 = std::min(vi.y2, vo.y2);

    int w = std::max(0, xx2 - xx1);
    int h = std::max(0, yy2 - yy1);

    int area = w * h;

    float dist = float(area) / float((vi.x2 - vi.x1) * (vi.y2 - vi.y1) +
                                     (vo.y2 - vo.y1) * (vo.x2 - vo.x1) - area);

    return dist;
}

static int non_max_suppression(std::vector<nv::detection_t> &detections,
                               const float iou_threshold) {
    // sort
    std::sort(detections.begin(), detections.end(),
              [](const nv::detection_t &d1, const nv::detection_t &d2) {
                  return d1.score > d2.score;
              });

    // nms
    std::vector<nv::detection_t> keep_detections;
    bool *suppressed = new bool[detections.size()];
    memset(suppressed, 0, sizeof(bool) * detections.size());
    const int num_detections = detections.size();
    for (int i = 0; i < num_detections; ++i) {
        if (suppressed[i])
            continue;
        keep_detections.emplace_back(detections[i]);
        for (int j = i + 1; j < num_detections; ++j) {
            if (suppressed[j])
                continue;
            float iou = bbox_overlap(detections[i].bbox, detections[j].bbox);
            if (iou > iou_threshold)
                suppressed[j] = true;
        }
    }
    keep_detections.swap(detections);
    delete[] suppressed;

    return 0;
}