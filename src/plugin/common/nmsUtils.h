/*** 
 * @Author: xingwg
 * @Date: 2024-10-12 09:59:37
 * @LastEditTime: 2024-10-12 10:58:10
 * @FilePath: /dmnn2/src/plugin/common/nmsUtils.h
 * @Description: 
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved. 
 */
#ifndef TRT_NMS_UTILS_H
#define TRT_NMS_UTILS_H

#include "plugin.h"

using namespace nvinfer1;
using namespace nvinfer1::plugin;

size_t detectionInferenceWorkspaceSize(bool shareLocation, int N, int C1, int C2, int numClasses, int numPredsPerClass, int topK, DataType DT_BBOX, DataType DT_SCORE);
size_t detectionInferenceWorkspaceSize(bool shareLocation, int N, int C1, int C2, int C3, int numClasses, int numPredsPerClass, int numPts, int topK, DataType DT_BBOX, DataType DT_SCORE, DataType DT_PTS);
size_t detectionInferenceWorkspaceSize(int N, int C1, int C2, int numClasses, int topK, DataType DT_BBOX, DataType DT_SCORE);
size_t detectionInferenceWorkspaceSize(int N, int C1, int C2, int numClasses, int numPts, int topK, DataType DT_BBOX, DataType DT_SCORE);

#endif
