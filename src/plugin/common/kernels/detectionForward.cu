/*
 * @Author: xingwg
 * @Date: 2024-10-12 09:59:37
 * @LastEditTime: 2024-10-12 11:58:41
 * @FilePath: /dmnn2/src/plugin/common/kernels/detectionForward.cu
 * @Description: 
 * 
 * Copyright (c) 2024 by Chinasvt, All Rights Reserved. 
 */
#include "kernel.h"
#include "../bboxUtils.h"
#include "error_check.h"

pluginStatus_t detectionInference(
        cudaStream_t stream,
        const int N,
        const int C1,
        const int C2,
        const bool shareLocation,
        const bool varianceEncodedInTarget,
        const int backgroundLabelId,
        const int numPredsPerClass,
        const int numClasses,
        const int topK,
        const int keepTopK,
        const float confidenceThreshold,
        const float nmsThreshold,
        const CodeTypeSSD codeType,
        const DataType DT_BBOX,
        const void* locData,
        const void* priorData,
        const DataType DT_SCORE,
        const void* confData,
        void* keepCount,
        void* topDetections,
        void* workspace,
        bool isNormalized,
        bool confSigmoid)
{
    // Batch size * number bbox per sample * 4 = total number of bounding boxes * 4
    const int locCount = N * C1;
    // Do not clip the bounding box that goes outside the image
    const bool clipBBox = false;
    /*
     * shareLocation
     * Bounding box are shared among all classes, i.e., a bounding box could be classified as any candidate class.
     * Otherwise
     * Bounding box are designed for specific classes, i.e., a bounding box could be classified as one certain class or not (binary classification).
     */
    const int numLocClasses = shareLocation ? 1 : numClasses;

    size_t bboxDataSize = detectionForwardBBoxDataSize(N, C1, DataType::kFLOAT);
    void* bboxDataRaw = workspace;

    pluginStatus_t status = decodeBBoxes(stream,
                                         locCount,
                                         codeType,
                                         varianceEncodedInTarget,
                                         numPredsPerClass,
                                         shareLocation,
                                         numLocClasses,
                                         backgroundLabelId,
                                         clipBBox,
                                         DataType::kFLOAT,
                                         locData,
                                         priorData,
                                         bboxDataRaw);
    if (STATUS_SUCCESS != status) {
        LOG_ERROR("Failed to decodeBBoxes");
        return status;
    }

    /*
     * bboxDataRaw format:
     * [batch size, numPriors (per sample), numLocClasses, 4]
     */
    // float for now
    void* bboxData;
    size_t bboxPermuteSize = detectionForwardBBoxPermuteSize(shareLocation, N, C1, DataType::kFLOAT);
    void* bboxPermute = nextWorkspacePtr((int8_t*) bboxDataRaw, bboxDataSize);

    /*
     * After permutation, bboxData format:
     * [batch_size, numLocClasses, numPriors (per sample) (numPredsPerClass), 4]
     * This is equivalent to swapping axis
     */
    if (!shareLocation)
    {
        status = permuteData(stream,
                             locCount,
                             numLocClasses,
                             numPredsPerClass,
                             4,
                             DataType::kFLOAT,
                             false,
                             bboxDataRaw,
                             bboxPermute);
        if (STATUS_SUCCESS != status) {
            LOG_ERROR("Failed to permuteData");
            return status;
        }
        bboxData = bboxPermute;
    }
        /*
         * If shareLocation, numLocClasses = 1
         * No need to permute data on linear memory
         */
    else
    {
        bboxData = bboxDataRaw;
    }
    /*
     * Conf data format
     * [batch size, numPriors * param.numClasses, 1, 1]
     */
    const int numScores = N * C2;
    size_t scoresSize = detectionForwardPreNMSSize(N, C2);
    void* scores = nextWorkspacePtr((int8_t*) bboxPermute, bboxPermuteSize);
    // need a conf_scores
    /*
     * After permutation, bboxData format:
     * [batch_size, numClasses, numPredsPerClass, 1]
     */
    status = permuteData(stream,
                         numScores,
                         numClasses,
                         numPredsPerClass,
                         1,
                         DataType::kFLOAT,
                         confSigmoid,
                         confData,
                         scores);
    if (STATUS_SUCCESS != status) {
        LOG_ERROR("Failed to permuteData");
        return status;
    }

    size_t indicesSize = detectionForwardPreNMSSize(N, C2);
    void* indices = nextWorkspacePtr((int8_t*) scores, scoresSize);

    size_t postNMSScoresSize = detectionForwardPostNMSSize(N, numClasses, topK);
    size_t postNMSIndicesSize = detectionForwardPostNMSSize(N, numClasses, topK);
    void* postNMSScores = nextWorkspacePtr((int8_t*) indices, indicesSize);
    void* postNMSIndices = nextWorkspacePtr((int8_t*) postNMSScores, postNMSScoresSize);

    //size_t sortingWorkspaceSize = sortScoresPerClassWorkspaceSize(N, numClasses, numPredsPerClass, FLOAT32);
    void* sortingWorkspace = nextWorkspacePtr((int8_t*) postNMSIndices, postNMSIndicesSize);
    // Sort the scores so that the following NMS could be applied.
    status = sortScoresPerClass(stream,
                                N,
                                numClasses,
                                numPredsPerClass,
                                backgroundLabelId,
                                confidenceThreshold,
                                DataType::kFLOAT,
                                scores,
                                indices,
                                sortingWorkspace);
    if (STATUS_SUCCESS != status) {
        LOG_ERROR("Failed to sortScoresPerClass");
        return status;
    }

    // NMS
    status = allClassNMS(stream,
                         N,
                         numClasses,
                         numPredsPerClass,
                         topK,
                         nmsThreshold,
                         shareLocation,
                         isNormalized,
                         DataType::kFLOAT,
                         DataType::kFLOAT,
                         bboxData,
                         scores,
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
                                N,
                                numClasses * topK,
                                DataType::kFLOAT,
                                postNMSScores,
                                postNMSIndices,
                                scores,
                                indices,
                                sortingWorkspace);
    if (STATUS_SUCCESS != status) {
        LOG_ERROR("Failed to sortScoresPerImage");
        return status;
    }

    // Gather data from the sorted bounding boxes after NMS
    status = gatherTopDetections(stream,
                                 shareLocation,
                                 true,
                                 N,
                                 numPredsPerClass,
                                 numClasses,
                                 topK,
                                 keepTopK,
                                 DataType::kFLOAT,
                                 DataType::kFLOAT,
                                 indices,
                                 scores,
                                 bboxData,
                                 keepCount,
                                 topDetections);
    if (STATUS_SUCCESS != status) {
        LOG_ERROR("Failed to gatherTopDetections");
        return status;
    }
    return STATUS_SUCCESS;
}

