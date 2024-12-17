/*
 * @Author: xingwg
 * @Date: 2024-10-12 09:59:37
 * @LastEditTime: 2024-10-12 11:50:22
 * @FilePath: /dmnn2/src/plugin/YOLONMSPluginV2/YOLONMSV2.cu
 * @Description: 
 * 
 * Copyright (c) 2024 by Chinasvt, All Rights Reserved. 
 */
#include "YOLONMSPluginV2.h"
#include "../common/bboxUtils.h"

using namespace nvinfer1::plugin;

__global__ void parseBBoxes(const int nbThreads, const int numClasses, const int numPts, const int C2, const float conf_thresh,
                            const float *inputData, float *bboxData, float *landData, float *confData) {
    CUDA_KERNEL_LOOP(idx, nbThreads) {
        float obj_conf = inputData[idx * C2 + 14];
        float max_score = 0;
        for (int c = 0; c < numClasses; ++c) {
            float score = inputData[idx * C2 + 15 + c] * obj_conf;
            if (score >= max_score)
                max_score = score;
            confData[idx * numClasses + c] = score;
        }

        if (max_score < conf_thresh)
            return;

        bboxData[idx * 4 + 0] = inputData[idx * C2 + 0] - inputData[idx * C2 + 2] * 0.5f;
        bboxData[idx * 4 + 1] = inputData[idx * C2 + 1] - inputData[idx * C2 + 3] * 0.5f;
        bboxData[idx * 4 + 2] = inputData[idx * C2 + 0] + inputData[idx * C2 + 2] * 0.5f;
        bboxData[idx * 4 + 3] = inputData[idx * C2 + 1] + inputData[idx * C2 + 3] * 0.5f;

        const int step = 2 * numPts;
        for (int n=0; n<numPts; ++n) {
            landData[idx * step + 0 + 2 * n] = inputData[idx * C2 + 4 + 2 * n];
            landData[idx * step + 1 + 2 * n] = inputData[idx * C2 + 5 + 2 * n];
        }
    }
}

void YOLODetectionInference(
        cudaStream_t stream,
        const int N,
        const int C1,  // C1
        const int C2,  // 4 + 1 + num_cls
        const int numClasses,
        const int numPts,
        const int keep_topK,
        const int topK,
        const float nms_thresh,
        const float conf_thresh,
        const float *inputData,
        void *workspace,
        float *topDetections) {

    size_t bboxDataSize = detectionForwardBBoxDataSize(N, C1 * 4, DataType::kFLOAT);
    void *bboxData = workspace;   // batch * C1 * 4 * 1

    size_t landDataSize = detectionForwardPtsDataSize(N, C1 * 2 * numPts, DataType::kFLOAT);
    void *landData = nextWorkspacePtr((int8_t *) bboxData, bboxDataSize);  // batch * C1 * 2 * numPts * 1

    size_t confDataSize = detectionForwardConfDataSize(N, C1 * numClasses, DataType::kFLOAT);
    void *confData = nextWorkspacePtr((int8_t *) landData, landDataSize);   // batch * C1 * num_cls * 1

    const int nbThreads = N * C1;
    parseBBoxes<<< CUDA_GET_BLOCKS(nbThreads), CUDA_NUM_THREADS, 0, stream>>>
            (nbThreads, numClasses, numPts, C2, conf_thresh, inputData, (float *) bboxData, (float *) landData, (float *) confData);

    // Conf data format
    // [batch_size, C1, num_cls, 1]
    const int num_scores = N * C1 * numClasses;
    size_t scoresSize = detectionForwardPreNMSSize(N, C1 * numClasses);
    void *scores = nextWorkspacePtr((int8_t *) confData, confDataSize);
    // need a conf_scores
    // After permutation, bboxData format:
    // [batch_size, num_cls, C1, 1]
    pluginStatus_t status = permuteData(stream,
                                        num_scores,
                                        numClasses,
                                        C1,
                                        1,
                                        DataType::kFLOAT,
                                        false,
                                        confData,
                                        scores);
    assert(status == STATUS_SUCCESS);

    size_t indicesSize = detectionForwardPreNMSSize(N, C1 * numClasses);
    void *indices = nextWorkspacePtr((int8_t *) scores, scoresSize);

    size_t postNMSScoresSize = detectionForwardPostNMSSize(N, numClasses, topK);
    void *postNMSScores = nextWorkspacePtr((int8_t *) indices, indicesSize);

    size_t postNMSIndicesSize = detectionForwardPostNMSSize(N, numClasses, topK);
    void *postNMSIndices = nextWorkspacePtr((int8_t *) postNMSScores, postNMSScoresSize);

    void *sortingWorkspace = nextWorkspacePtr((int8_t *) postNMSIndices, postNMSIndicesSize);
    // Sort the scores so that the following NMS could be applied.
    status = sortScoresPerClass(stream,
                                N,
                                numClasses,
                                C1,
                                -1,  // no background_id
                                conf_thresh,
                                DataType::kFLOAT,
                                scores,
                                indices,
                                sortingWorkspace);
    assert(status == STATUS_SUCCESS);

    // NMS
    status = allClassNMS(stream,
                         N,
                         numClasses,
                         C1,
                         topK,
                         nms_thresh,
                         true,
                         false,
                         DataType::kFLOAT,
                         DataType::kFLOAT,
                         bboxData,
                         scores,
                         indices,
                         postNMSScores,
                         postNMSIndices,
                         false);
    assert(status == STATUS_SUCCESS);

    // Sort the bounding boxes after NMS using scores
    status = sortScoresPerImage(stream,
                                N,
                                numClasses * topK,
                                DataType::kFLOAT,
                                postNMSScores,
                                postNMSIndices,
                                scores,
                                indices,
                                sortingWorkspace);
    assert(status == STATUS_SUCCESS);

    // Gather data from the sorted bounding boxes after NMS
    status = gatherTopDetectionsV2(stream,
                                   true,
                                   N,
                                   C1,
                                   numClasses,
                                   numPts,
                                   topK,
                                   keep_topK,
                                   DataType::kFLOAT,
                                   DataType::kFLOAT,
                                   indices,
                                   scores,
                                   bboxData,
                                   landData,
                                   nullptr,
                                   topDetections);
    assert(status == STATUS_SUCCESS);
}

int YOLODetectionOutputV2::enqueue(int batchSize, void const *const *inputs, void *const *outputs, void *workspace,
                                   cudaStream_t stream) noexcept {
    const auto *predData = reinterpret_cast<const float *>(inputs[0]);  // N*C1*C2*1
    auto *topDetections = reinterpret_cast<float *>(outputs[0]);
    YOLODetectionInference(stream, batchSize, C1, C2, param.num_classes, param.num_pts, param.keep_topK, param.topK, param.nms_thresh,
                           param.conf_thresh, predData, workspace, topDetections);
    return 0;
}

int YOLODetectionOutputDynamicV2::enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
                                          const nvinfer1::PluginTensorDesc *outputDesc,
                                          const void *const *inputs, void *const *outputs, void *workspace,
                                          cudaStream_t stream) noexcept {
    const auto *predData = reinterpret_cast<const float *>(inputs[0]);  // N*C1*C2*1
    auto *topDetections = reinterpret_cast<float *>(outputs[0]);
    int batchSize = inputDesc[0].dims.d[0];
    YOLODetectionInference(stream, batchSize, C1, C2, param.num_classes, param.num_pts, param.keep_topK, param.topK, param.nms_thresh,
                           param.conf_thresh, predData, workspace, topDetections);
    return 0;
}