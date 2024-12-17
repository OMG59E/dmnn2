/*
 * @Author: xingwg
 * @Date: 2024-10-12 09:59:37
 * @LastEditTime: 2024-10-12 11:51:39
 * @FilePath: /dmnn2/src/plugin/YOLOXNMSPlugin/YOLOXNMS.cu
 * @Description: 
 * 
 * Copyright (c) 2024 by Chinasvt, All Rights Reserved. 
 */

#include "YOLOXNMSPlugin.h"
#include "../common/bboxUtils.h"

using namespace nvinfer1::plugin;

__global__ void parseYOLOXBBoxes(const int nbThreads, const int numClasses, const int C1, const int C2,
        const float conf_thresh, const int3 layer_sizes, const int3 strides,
        const float *inputData, float *bboxData, float *confData ) {
    CUDA_KERNEL_LOOP(idx, nbThreads) {
        int dn = idx / C1;
        int anchor_size0 = layer_sizes.x * layer_sizes.x;
        int anchor_size1 = layer_sizes.y * layer_sizes.y;
        int new_idx = idx - dn * C1;
        int grid_x = 0, grid_y = 0, stride = 0;
        if (new_idx < anchor_size0) {
            grid_x = new_idx % layer_sizes.x;
            grid_y = new_idx / layer_sizes.x;
            stride = strides.x;
        } else if (new_idx < (anchor_size0 + anchor_size1)) {
            new_idx -= anchor_size0;
            grid_x = new_idx % layer_sizes.y;
            grid_y = new_idx / layer_sizes.y;
            stride = strides.y;
        } else {
            new_idx -= (anchor_size0 + anchor_size1);
            grid_x = new_idx % layer_sizes.z;
            grid_y = new_idx / layer_sizes.z;
            stride = strides.z;
        }

        float cx = (inputData[idx * C2 + 0] + grid_x) * stride;
        float cy = (inputData[idx * C2 + 1] + grid_y) * stride;
        float w = exp(inputData[idx * C2 + 2]) * stride;
        float h = exp(inputData[idx * C2 + 3]) * stride;

        bboxData[idx * 4 + 0] = cx - w * 0.5f;
        bboxData[idx * 4 + 1] = cy - h * 0.5f;
        bboxData[idx * 4 + 2] = cx + w * 0.5f;
        bboxData[idx * 4 + 3] = cy + h * 0.5f;

        for (int c=0; c<numClasses; c++)
            confData[idx * numClasses + c] = inputData[idx * C2 + 5 + c] * inputData[idx * C2 + 4];
    }
}

__global__ void parseYOLOXBBoxes(const int nbThreads, const int numClasses, const int C1, const int C2,
        const float conf_thresh, const int4 layer_sizes, const int4 strides,
        const float *inputData, float *bboxData, float *confData ) {
    CUDA_KERNEL_LOOP(idx, nbThreads) {
        int dn = idx / C1;
        int anchor_size0 = layer_sizes.x * layer_sizes.x;
        int anchor_size1 = layer_sizes.y * layer_sizes.y;
        int anchor_size2 = layer_sizes.z * layer_sizes.z;
        int new_idx = idx - dn * C1;
        int dx = 0, dy = 0, stride = 0;
        if (new_idx < anchor_size0) {
            dx = new_idx % layer_sizes.x;
            dy = new_idx / layer_sizes.x;
            stride = strides.x;
        } else if (new_idx < anchor_size0 + anchor_size1) {
            dx = new_idx % layer_sizes.y;
            dy = new_idx / layer_sizes.y;
            stride = strides.y;
        } else if (new_idx < anchor_size0 + anchor_size1 + anchor_size2) {
            dx = new_idx % layer_sizes.z;
            dy = new_idx / layer_sizes.z;
            stride = strides.z;
        } else {
            dx = new_idx % layer_sizes.w;
            dy = new_idx / layer_sizes.w;
            stride = strides.w;
        }

        float cx = (inputData[idx * C2 + 0] + dx) * stride;
        float cy = (inputData[idx * C2 + 1] + dy) * stride;
        float w = (inputData[idx * C2 + 2] + dx) * stride;
        float h = (inputData[idx * C2 + 3] + dy) * stride;

        bboxData[idx * 4 + 0] = cx - w * 0.5f;
        bboxData[idx * 4 + 1] = cy - h * 0.5f;
        bboxData[idx * 4 + 2] = cx + w * 0.5f;
        bboxData[idx * 4 + 3] = cy + h * 0.5f;

        float score = 0;
        for (int c=0; c<numClasses; c++) {
            score = inputData[idx * C2 + 5 + c] * inputData[idx * C2 + 4];
            confData[idx * numClasses + c] = score < conf_thresh ? 0 : score;
        }
    }
}

void YOLOXDetectionInference(
        cudaStream_t stream,
        const int N,
        const int C1,  // C1
        const int C2,  // 4 + 1 + num_cls
        const int numClasses,
        const int keep_topK,
        const int topK,
        bool use_p6,
        const float nms_thresh,
        const float conf_thresh,
        const float *inputData,
        void* workspace,
        float *topDetections) {
    size_t bboxDataSize = detectionForwardBBoxDataSize(N, C1 * 4, DataType::kFLOAT);
    void* bboxData = workspace;   // batch * C1 * 4 * 1

    size_t confDataSize = detectionForwardConfDataSize(N, C1 * numClasses, DataType::kFLOAT);
    void* confData = nextWorkspacePtr((int8_t*) bboxData, bboxDataSize);   // batch * C1 * num_cls * 1

    const int nbThreads = N * C1;
    if (use_p6) {
        int step = int(std::sqrt(C1 / 85));
        const int4 strides = make_int4(8, 16, 32, 64);
        const int4 layer_sizes = make_int4(8*step, 4*step, 2*step, step);
        parseYOLOXBBoxes<< < CUDA_GET_BLOCKS(nbThreads), CUDA_NUM_THREADS, 0, stream>> >(nbThreads,
                numClasses, C1, C2, conf_thresh, layer_sizes, strides, inputData, (float*)bboxData, (float*)confData);
    } else {
        int step = int(std::sqrt(C1 / 21));
        const int3 strides = make_int3(8, 16, 32);
        const int3 layer_sizes = make_int3(4*step, 2*step, step);
        parseYOLOXBBoxes<< < CUDA_GET_BLOCKS(nbThreads), CUDA_NUM_THREADS, 0, stream>> >(nbThreads,
                numClasses, C1, C2, conf_thresh, layer_sizes, strides, inputData, (float*)bboxData, (float*)confData);
    }

    // Conf data format
    // [batch_size, C1, num_cls, 1]
    const int num_scores = N * C1 * numClasses;
    size_t scoresSize = detectionForwardPreNMSSize(N, C1 * numClasses);
    void* scores = nextWorkspacePtr((int8_t*) confData, confDataSize);
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
    void* indices = nextWorkspacePtr((int8_t*) scores, scoresSize);

    size_t postNMSScoresSize = detectionForwardPostNMSSize(N, numClasses, topK);
    void* postNMSScores = nextWorkspacePtr((int8_t*) indices, indicesSize);

    size_t postNMSIndicesSize = detectionForwardPostNMSSize(N, numClasses, topK);
    void* postNMSIndices = nextWorkspacePtr((int8_t*) postNMSScores, postNMSScoresSize);

    void* sortingWorkspace = nextWorkspacePtr((int8_t*) postNMSIndices, postNMSIndicesSize);
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
    status = gatherTopDetections(stream,
                                 true,
                                 false,
                                 N,
                                 C1,
                                 numClasses,
                                 topK,
                                 keep_topK,
                                 DataType::kFLOAT,
                                 DataType::kFLOAT,
                                 indices,
                                 scores,
                                 bboxData,
                                 nullptr,
                                 topDetections);

    assert(status == STATUS_SUCCESS);
}

int YOLOXDetectionOutput::enqueue(int batchSize, void const *const *inputs,
        void *const*outputs, void *workspace, cudaStream_t stream) noexcept {
    const auto *predData = reinterpret_cast<const float*>(inputs[0]);  // N*C1*C2*1
    auto *topDetections = reinterpret_cast<float*>(outputs[0]);
    YOLOXDetectionInference(stream, batchSize, C1, C2, param.num_classes, param.keep_topK, param.topK, param.use_p6,
                            param.nms_thresh, param.conf_thresh, predData, workspace, topDetections);
    return 0;
}