/*
 * @Author: xingwg
 * @Date: 2024-10-12 09:59:37
 * @LastEditTime: 2024-10-12 11:03:23
 * @FilePath: /dmnn2/src/plugin/ctNMSPlugin/ctNMS.cu
 * @Description: 
 * 
 * Copyright (c) 2024 by Chinasvt, All Rights Reserved. 
 */
#include <cub/cub.cuh>
#include <algorithm>
#include <iostream>
#include <math.h>
#include <vector>
#include "ctNMSPlugin.h"
#include "../common/cub_helper.h"
#include "../common/bboxUtils.h"
#include "../common/kernels/kernel.h"
#include "error_check.h"

__global__ void ctdecode_kernel(
        const int nbThreads,
        int H,
        int W,
        int num_pts,
        float conf_thresh,
        const float* hmData,
        const float* scaleData,
        const float* offsetData,
        const float* landDataRaw,
        float* bboxData,
        float* scoreData,
        float* landData) {
    CUDA_KERNEL_LOOP(idx, nbThreads) {
        int dw = idx % W;  // 0-160
        int dh = idx / W % H; // 0-160
        int dn = idx / W / H;

        float inputSizeW = 4.0f * W;
        float inputSizeH = 4.0f * H;

        float scale_w = exp(scaleData[(dn * 2 * H * W) + (1 * H * W) + (dh * W) + dw]) * 4;
        float scale_h = exp(scaleData[(dn * 2 * H * W) + (0 * H * W) + (dh * W) + dw]) * 4;

        float offset_w = offsetData[(dn * 2 * H * W) + (1 * H * W) + (dh * W) + dw];
        float offset_h = offsetData[(dn * 2 * H * W) + (0 * H * W) + (dh * W) + dw];

        float x1 = min(max(0.0f, (dw + offset_w + 0.5f) * 4 - scale_w * 0.5f), inputSizeW);
        float y1 = min(max(0.0f, (dh + offset_h + 0.5f) * 4 - scale_h * 0.5f), inputSizeH);
        float x2 = min(x1 + scale_w, inputSizeW);
        float y2 = min(y1 + scale_h, inputSizeH);
        float score = hmData[idx];
        if (score < conf_thresh)
            score = 0;

        bboxData[4 * idx + 0] = x1 / inputSizeW;
        bboxData[4 * idx + 1] = y1 / inputSizeH;
        bboxData[4 * idx + 2] = x2 / inputSizeW;
        bboxData[4 * idx + 3] = y2 / inputSizeH;

        scoreData[idx] = score;

        int step = 2 * num_pts;
        for (int k=0; k<num_pts; ++k) {
            float px = landDataRaw[(dn * 2 * num_pts * H * W) + ((2 * k + 1) * H * W) + (dh * W) + dw] * scale_w + x1;
            float py = landDataRaw[(dn * 2 * num_pts * H * W) + ((2 * k + 0) * H * W) + (dh * W) + dw] * scale_h + y1;
            landData[step * idx + 2 * k + 0] = px / inputSizeW;
            landData[step * idx + 2 * k + 1] = py / inputSizeH;
        }
    }
}

pluginStatus_t CenterFaceOutput::ctFaceNMSInference(cudaStream_t stream, int batch, const float *hmData,
                                          const float *scaleData, const float *offsetData, const float *landDataRaw,
                                          void *workspace, float *topDetections) {
    const int locCount = batch * H_ * W_;
    void* bboxData = workspace;
    void* scoreData = nextWorkspacePtr((int8_t*)bboxData, locCount * 4 * sizeof(float));
    void* landData = nextWorkspacePtr((int8_t*)scoreData, locCount * 1 * sizeof(float));
    ctdecode_kernel<<<CUDA_GET_BLOCKS(locCount), CUDA_NUM_THREADS, 0, stream>>>(locCount, H_, W_,
            param_.num_pts, param_.confidence_threshold, hmData, scaleData, offsetData, landDataRaw,
            (float*)bboxData, (float*)scoreData, (float*)landData);

    size_t indicesSize = detectionForwardPreNMSSize(batch, H_ * W_ * param_.num_classes);
    void* indices = nextWorkspacePtr((int8_t*)landData, locCount*2*param_.num_pts*sizeof(float));

    size_t postNMSScoresSize = detectionForwardPostNMSSize(batch, param_.num_classes, param_.topK);
    void* postNMSScores = nextWorkspacePtr((int8_t*)indices, indicesSize);

    size_t postNMSIndicesSize = detectionForwardPostNMSSize(batch, param_.num_classes, param_.topK);
    void* postNMSIndices = nextWorkspacePtr((int8_t*)postNMSScores, postNMSScoresSize);

    void* sortingWorkspace = nextWorkspacePtr((int8_t*)postNMSIndices, postNMSIndicesSize);

    pluginStatus_t status = sortScoresPerClass(stream,
                                               batch,
                                               param_.num_classes,
                                               H_*W_,
                                               -1,
                                               param_.confidence_threshold,
                                               DataType::kFLOAT,
                                               scoreData,
                                               indices,
                                               sortingWorkspace);
    if (STATUS_SUCCESS != status) {
        LOG_ERROR("Failed to sortScoresPerClass");
        return status;
    }

    // NMS
    status = allClassNMS(stream,
                         batch,
                         param_.num_classes,
                         H_*W_,
                         param_.topK,
                         param_.nms_threshold,
                         true,
                         true,
                         DataType::kFLOAT,
                         DataType::kFLOAT,
                         bboxData,
                         scoreData,
                         indices,
                         postNMSScores,
                         postNMSIndices,
                         false);
    if (STATUS_SUCCESS != status) {
        LOG_ERROR("Failed to allClassNMS");
        return status;
    }

    // Sort the bounding boxes after NMS using scores
    status = sortScoresPerImage(stream,
                                batch,
                                param_.num_classes * param_.topK,
                                DataType::kFLOAT,
                                postNMSScores,
                                postNMSIndices,
                                scoreData,
                                indices,
                                sortingWorkspace);
    if (STATUS_SUCCESS != status) {
        LOG_ERROR("Failed to sortScoresPerImage");
        return status;
    }

    status = gatherTopDetectionsV2(stream,
                                   true,
                                   batch,
                                   H_*W_,
                                   param_.num_classes,
                                   param_.num_pts,
                                   param_.topK,
                                   param_.keep_topK,
                                   DataType::kFLOAT,
                                   DataType::kFLOAT,
                                   indices,
                                   scoreData,
                                   bboxData,
                                   landData,
                                   nullptr,
                                   topDetections);
    if (STATUS_SUCCESS != status) {
        LOG_ERROR("Failed to gatherTopDetectionsV2");
        return status;
    }
    return STATUS_SUCCESS;
}

int CenterFaceOutput::enqueue(int32_t batchSize, void const* const* inputs, void* const* outputs, void *workspace, cudaStream_t stream) noexcept {
    const auto heatmap_data = reinterpret_cast<const float*>(inputs[0]);
    const auto scale_data = reinterpret_cast<const float*>(inputs[1]);
    const auto offset_data = reinterpret_cast<const float*>(inputs[2]);
    const auto landmark_data = reinterpret_cast<const float*>(inputs[3]);
    auto topDetections = reinterpret_cast<float*>(outputs[0]);
    ctFaceNMSInference(stream, batchSize, heatmap_data, scale_data, offset_data, landmark_data, workspace, topDetections);
    return 0;
}