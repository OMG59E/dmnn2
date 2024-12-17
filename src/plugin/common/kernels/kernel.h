/*** 
 * @Author: xingwg
 * @Date: 2024-10-12 09:59:37
 * @LastEditTime: 2024-10-12 10:36:26
 * @FilePath: /dmnn2/src/plugin/common/kernels/kernel.h
 * @Description: 
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved. 
 */

#ifndef TRT_KERNEL_H
#define TRT_KERNEL_H

#include "error_check.h"
#include "../plugin.h"

#include <cublas_v2.h>
#include <cassert>
#include <cstdio>

#ifndef AARCH64
#include <nvjpeg.h>
#endif

using namespace nvinfer1;
using namespace nvinfer1::plugin;

//#define DEBUG_ENABLE 0

typedef enum {
    NCHW = 0,
    NC4HW = 1
} DLayout_t;

pluginStatus_t allClassNMS(cudaStream_t stream, int num, int num_classes, int num_preds_per_class, int top_k,
                           float nms_threshold, bool share_location, bool isNormalized, DataType DT_SCORE,
                           DataType DT_BBOX, void *bbox_data,
                           void *beforeNMS_scores, void *beforeNMS_index_array, void *afterNMS_scores,
                           void *afterNMS_index_array,
                           bool flipXY = false);

pluginStatus_t allClassNMSV2(cudaStream_t stream, int num, int num_classes, int num_preds_per_class, int top_k,
                             float nms_threshold, bool share_location, bool isNormalized, DataType DT_SCORE,
                             DataType DT_BBOX, void *bbox_data,
                             void *beforeNMS_scores, void *beforeNMS_index_array, void *afterNMS_scores,
                             void *afterNMS_index_array, void *pts_data,
                             void *beforeLKNMS_scores, void *beforeLKNMS_index_array, void *afterLKNMS_scores,
                             void *afterLKNMS_index_array,
                             bool flipXY = false);

pluginStatus_t detectionInference(cudaStream_t stream, int N, int C1, int C2, bool shareLocation,
                                  bool varianceEncodedInTarget, int backgroundLabelId, int numPredsPerClass,
                                  int numClasses, int topK, int keepTopK,
                                  float confidenceThreshold, float nmsThreshold, CodeTypeSSD codeType, DataType DT_BBOX,
                                  const void *locData,
                                  const void *priorData, DataType DT_SCORE, const void *confData, void *keepCount,
                                  void *topDetections,
                                  void *workspace, bool isNormalized = true, bool confSigmoid = false);

pluginStatus_t detectionInferenceV2(cudaStream_t stream, int N, int C1, int C2, int C3,
                                    bool shareLocation,
                                    bool varianceEncodedInTarget, int backgroundLabelId,
                                    int numPredsPerClass, int numClasses, int numPts, int topK,
                                    int keepTopK,
                                    float confidenceThreshold, const float nmsThreshold,
                                    CodeTypeSSD codeType, const DataType DT_BBOX, const void *locData,
                                    const void *priorData, const void *ptData, DataType DT_SCORE,
                                    const void *confData, void *keepCount, void *topDetections,
                                    void *workspace, bool isNormalized = true, bool confSigmoid = false
);

pluginStatus_t
gatherTopDetections(cudaStream_t stream, bool shareLocation, bool useClip, int numImages, int numPredsPerClass,
                    int numClasses, int topK, int keepTopK, DataType DT_BBOX, DataType DT_SCORE, const void *indices,
                    const void *scores, const void *bboxData, void *keepCount, void *topDetections);

pluginStatus_t gatherTopDetectionsV2(cudaStream_t stream, bool shareLocation, int numImages, int numPredsPerClass,
                                     int numClasses, int numPts, int topK, int keepTopK, DataType DT_BBOX,
                                     DataType DT_SCORE,
                                     const void *indices,
                                     const void *scores, const void *bboxData, const void *ptsData,
                                     void *keepCount, void *topLKDetections);

size_t detectionForwardBBoxDataSize(int N, int C1, DataType DT_BBOX);

size_t detectionForwardConfDataSize(int N, int C1, DataType DT_BBOX);

size_t detectionForwardBBoxPermuteSize(bool shareLocation, int N, int C1, DataType DT_BBOX);

size_t sortScoresPerClassWorkspaceSize(int num, int num_classes, int num_preds_per_class, DataType DT_CONF);

size_t sortScoresPerImageWorkspaceSize(int num_images, int num_items_per_image, DataType DT_SCORE);

pluginStatus_t sortScoresPerImage(cudaStream_t stream, int num_images, int num_items_per_image, DataType DT_SCORE,
                                  void *unsorted_scores, void *unsorted_bbox_indices, void *sorted_scores,
                                  void *sorted_bbox_indices,
                                  void *workspace);

pluginStatus_t sortScoresPerClass(cudaStream_t stream, int num, int num_classes, int num_preds_per_class,
                                  int background_label_id, float confidence_threshold, DataType DT_SCORE,
                                  void *conf_scores_gpu,
                                  void *index_array_gpu, void *workspace);


size_t calculateTotalWorkspaceSize(size_t *workspaces, int count);

pluginStatus_t permuteData(cudaStream_t stream, int nthreads, int num_classes, int num_data, int num_dim,
                           DataType DT_DATA, bool confSigmoid, const void *data, void *new_data);

size_t detectionForwardPreNMSSize(int N, int C2);

size_t detectionForwardPtsPermuteSize(bool shareLocation, int N, int C3, DataType DT_BBOX);

size_t detectionForwardPtsDataSize(int N, int C3, DataType DT_PTS);

size_t detectionForwardPostNMSSize(int N, int numClasses, int topK);

pluginStatus_t decodeBBoxes(cudaStream_t stream, int nthreads, CodeTypeSSD code_type, bool variance_encoded_in_target,
                            int num_priors, bool share_location, int num_loc_classes, int background_label_id,
                            bool clip_bbox, DataType DT_BBOX,
                            const void *loc_data, const void *prior_data, void *bbox_data);

size_t normalizePluginWorkspaceSize(bool acrossSpatial, int C, int H, int W);

pluginStatus_t normalizeInference(cudaStream_t stream, cublasHandle_t handle, bool acrossSpatial, bool channelShared,
                                  int N, int C, int H, int W, float eps, const void *scale, const void *inputData,
                                  void *outputData, void *workspace);

pluginStatus_t priorBoxInference(cudaStream_t stream, PriorBoxParameters param, int H, int W, int numPriors,
                                 int numAspectRatios, const void *minSize, const void *maxSize,
                                 const void *aspectRatios, void *outputData);
#endif
