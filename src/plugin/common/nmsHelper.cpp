/*** 
 * @Author: xingwg
 * @Date: 2024-10-12 09:59:37
 * @LastEditTime: 2024-10-12 10:57:44
 * @FilePath: /dmnn2/src/plugin/common/nmsHelper.cpp
 * @Description: 
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved. 
 */
#include <algorithm>
#include "error_check.h"
#include "plugin.h"

using namespace nvinfer1;
using namespace nvinfer1::plugin;

size_t detectionForwardBBoxDataSize(int N, int C1, DataType DT_BBOX) {
    if (DT_BBOX == DataType::kFLOAT)
        return N * C1 * sizeof(float);
    LOG_INFO("Only FP32 type bounding boxes are supported.");
    return (size_t) -1;
}

size_t detectionForwardConfDataSize(int N, int C1, DataType DT_BBOX) {
    if (DT_BBOX == DataType::kFLOAT)
        return N * C1 * sizeof(float);
    LOG_INFO("Only FP32 type bounding boxes are supported.");
    return (size_t) -1;
}


size_t detectionForwardPtsDataSize(int N, int C3, DataType DT_PTS) {
    if (DT_PTS == DataType::kFLOAT)
        return N * C3 * sizeof(float);
    LOG_INFO("Only FP32 type pts are supported.");
    return (size_t) -1;
}

size_t detectionForwardPtsPermuteSize(bool shareLocation, int N, int C3, DataType DT_PTS) {
    if (DT_PTS == DataType::kFLOAT)
        return shareLocation ? 0 : N * C3 * sizeof(float);
    LOG_INFO("Only FP32 type pts are supported.");
    return (size_t) -1;
}

size_t detectionForwardBBoxPermuteSize(bool shareLocation, int N, int C1, DataType DT_BBOX) {
    if (DT_BBOX == DataType::kFLOAT)
        return shareLocation ? 0 : N * C1 * sizeof(float);
    LOG_INFO("Only FP32 type bounding boxes are supported.");
    return (size_t) -1;
}

size_t detectionForwardPreNMSSize(int N, int C2) { return N * C2 * sizeof(float); }
size_t detectionForwardPreLKNMSSize(int N, int C3) {  return N * C3 * sizeof(float); }
size_t detectionForwardPostNMSSize(int N, int numClasses, int topK) { return N * numClasses * topK * sizeof(float); }
