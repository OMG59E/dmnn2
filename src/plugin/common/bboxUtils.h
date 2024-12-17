/*** 
 * @Author: xingwg
 * @Date: 2024-10-12 09:59:37
 * @LastEditTime: 2024-10-12 10:53:59
 * @FilePath: /dmnn2/src/plugin/common/bboxUtils.h
 * @Description: 
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved. 
 */
#ifndef TRT_BBOX_UTILS_H
#define TRT_BBOX_UTILS_H

#include "plugin.h"

using namespace nvinfer1;
using namespace nvinfer1::plugin;

template<typename T>
struct Bbox {
    T xmin, ymin, xmax, ymax;
    Bbox(T xmin, T ymin, T xmax, T ymax)
            : xmin(xmin), ymin(ymin), xmax(xmax), ymax(ymax) {
    }
    Bbox() = default;
};


template<typename T>
struct BboxInfo {
    T conf_score;
    int label;
    int bbox_idx;
    bool kept;
    BboxInfo(T conf_score, int label, int bbox_idx, bool kept)
            : conf_score(conf_score), label(label), bbox_idx(bbox_idx), kept(kept) {
    }
    BboxInfo() = default;
};

template<typename TFloat>
bool operator<(const Bbox<TFloat> &lhs, const Bbox<TFloat> &rhs) { return lhs.x1 < rhs.x1; }

template<typename TFloat>
bool operator==(const Bbox<TFloat> &lhs, const Bbox<TFloat> &rhs) { return lhs.x1 == rhs.x1 && lhs.y1 == rhs.y1 && lhs.x2 == rhs.x2 && lhs.y2 == rhs.y2; }

int8_t *alignPtr(int8_t *ptr, uintptr_t to);
int8_t *nextWorkspacePtr(int8_t *ptr, uintptr_t previousWorkspaceSize);
size_t dataTypeSize(DataType dtype);
void setUniformOffsets(cudaStream_t stream, int num_segments, int offset, int *d_offsets);

template<typename T>
struct Pts {
    T l1_x, l1_y, l2_x, l2_y, l3_x, l3_y, l4_x, l4_y;
    Pts(T l1_x, T l1_y, T l2_x, T l2_y, T l3_x, T l3_y, T l4_x, T l4_y)
            : l1_x(l1_x), l1_y(l1_y), l2_x(l2_x), l2_y(l2_y), l3_x(l3_x), l3_y(l3_y), l4_x(l4_x), l4_y(l4_y) {

    }
    Pts() = default;
};

template<typename T>
struct PtsInfo {
    T conf_score;
    int label;
    int lk_idx;
    bool kept;
    PtsInfo(T conf_score, int label, int lk_idx, bool kept)
            : conf_score(conf_score), label(label), lk_idx(lk_idx), kept(kept) {
    }
    PtsInfo() = default;
};

#endif


