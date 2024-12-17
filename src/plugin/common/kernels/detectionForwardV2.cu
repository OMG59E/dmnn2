/*
 * @Author: xingwg
 * @Date: 2024-10-12 09:59:37
 * @LastEditTime: 2024-10-12 10:33:44
 * @FilePath: /dmnn2/src/plugin/common/kernels/detectionForwardV2.cu
 * @Description: 
 * 
 * Copyright (c) 2024 by Chinasvt, All Rights Reserved. 
 */
#include "kernel.h"
#include "../bboxUtils.h"
#include "error_check.h"
#include "base_types.h"

__global__ void decode_pts_kernel(
        const int nbThreads,
        const bool varianceEncodedInTarget,
        const int numPriors,
        const int numPts,
        const bool clipBBox,
        const float* landData,    // [batch_size, numPriors*numPts*2, 1, 1]
        const float* priorData,   // [batch_size, 2, numPriors*4, 1]
        float* ptsData)           // [batch_size, numPriors*numPts*2, 1, 1]
{
    CUDA_KERNEL_LOOP(idx, nbThreads)
    {
        int dn = idx / numPriors;   // 图片索引
        int dp = idx % numPriors;   // 先验框索引

        // Get prior box coordinates
        const float x1 = priorData[dp*4 + 0];
        const float y1 = priorData[dp*4 + 1];
        const float x2 = priorData[dp*4 + 2];
        const float y2 = priorData[dp*4 + 3];

        // Calculate prior box center, height, and width
        const float cx = (x1 + x2) * 0.5f;
        const float cy = (y1 + y2) * 0.5f;
        const float w = x2 - x1;
        const float h = y2 - y1;

        for (int i=0; i<numPts; i++)
        {
            if (varianceEncodedInTarget)
            {

            }
            else
            {
                ptsData[dn*numPriors*numPts*2 + dp*numPts*2 + i*2 + 0] = cx + landData[dn*numPriors*numPts*2 + dp*numPts*2 + i*2 + 0] * priorData[numPriors*4] * w;
                ptsData[dn*numPriors*numPts*2 + dp*numPts*2 + i*2 + 1] = cy + landData[dn*numPriors*numPts*2 + dp*numPts*2 + i*2 + 1] * priorData[numPriors*4] * h;
            }

            if (clipBBox)
            {
                ptsData[dn*numPriors*numPts*2 + dp*numPts*2 + i*2 + 0] = max(min(ptsData[dn*numPriors*numPts*2 + dp*numPts*2 + i*2 + 0], 1.0f), 0.0f);
                ptsData[dn*numPriors*numPts*2 + dp*numPts*2 + i*2 + 1] = max(min(ptsData[dn*numPriors*numPts*2 + dp*numPts*2 + i*2 + 1], 1.0f), 0.0f);
            }
        }
    }
}

pluginStatus_t detectionInferenceV2(
        cudaStream_t stream,
        const int N,
        const int C1,
        const int C2,
        const int C3,
        const bool shareLocation,
        const bool varianceEncodedInTarget,
        const int backgroundLabelId,
        const int numPredsPerClass,
        const int numClasses,
        const int numPts,
        const int topK,
        const int keepTopK,
        const float confidenceThreshold,
        const float nmsThreshold,
        const CodeTypeSSD codeType,
        const DataType DT_BBOX,
        const void *locData,
        const void *priorData,
        const void *landData,
        const DataType DT_SCORE,
        const void *confData,
        void *keepCount,
        void *topDetections,
        void *workspace,
        bool isNormalized,
        bool confSigmoid)
{
    const int locCount = N * C1;
    // Do not clip the bounding box that goes outside the image
    const bool clipBBox = false;
    /**
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

    /**
     * bboxDataRaw format:
     * [batch size, numPriors (per sample), numLocClasses, 4]
     */
    // float for now
    void* bboxData;
    size_t bboxPermuteSize = detectionForwardBBoxPermuteSize(shareLocation, N, C1, DataType::kFLOAT);
    void* bboxPermute = nextWorkspacePtr((int8_t*) bboxDataRaw, bboxDataSize);

    /**
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
    /**
     * If shareLocation, numLocClasses = 1
     * No need to permute data on linear memory
     */
    else
    {
        bboxData = bboxDataRaw;
    }

    /**
     * pts data format
     * [batch_size, numPriors*numPts*2, 1, 1]
     */
    //const int ptsCount = N * C3;
    const int nbThreads = N * numPredsPerClass;
    size_t ptsDataSize = detectionForwardPtsDataSize(N, C3, DataType::kFLOAT);
    void* ptsData = nextWorkspacePtr((int8_t *)bboxPermute, bboxPermuteSize);
    decode_pts_kernel<< < CUDA_GET_BLOCKS(nbThreads), CUDA_NUM_THREADS, 0, stream>> >
                (nbThreads, varianceEncodedInTarget, numPredsPerClass, numPts, clipBBox, (float*)landData, (float*)priorData, (float*)ptsData);

    /*
    pluginStatus_t status_pts = decodePts(stream,
                                          ptsCount,
                                          varianceEncodedInTarget,
                                          numPredsPerClass,
                                          numPts,
                                          clipBBox,
                                          DataType::kFLOAT,
                                          landData,
                                          priorData,
                                          ptsDataRaw);

    LOG_ASSERT(status_pts == STATUS_SUCCESS);

    void* ptsData;
    size_t ptsPermuteSize = detectionForwardPtsPermuteSize(shareLocation, N, C3, DataType::kFLOAT);
    void* ptsPermute = nextWorkspacePtr((int8_t*)ptsDataRaw, ptsDataSize);

    if(!shareLocation)
    {
        status_pts = permuteData(stream,
                                 ptsCount,
                                 numLocClasses,
                                 numPredsPerClass,
                                 2*numPts,
                                 DataType::kFLOAT,
                                 false,
                                 ptsDataRaw,
                                 ptsPermute);
        LOG_ASSERT(status_pts == STATUS_SUCCESS);
        ptsData = ptsPermute;
    }
    else
    {
        ptsData = ptsDataRaw;
    }
    */

    // 验证每个batch是否一致
    //float* dst{nullptr};
    //int count = N*C3;
    //int step = C3;
    //float* src = reinterpret_cast<float*>(ptsData);
    //CUDACHECK(cudaMallocHost((void**)&dst, count*sizeof(float)));
    //CUDACHECK(cudaMemcpy(dst, src, count*sizeof(float), cudaMemcpyDeviceToHost));
    //for (int i=1; i<count; i+=step)
    //    printf("%f\n", dst[i]);

    /**
     * Conf data format
     * [batch size, numPriors * param.numClasses, 1, 1]
     */
    const int numScores = N * C2;
    size_t scoresSize = detectionForwardPreNMSSize(N, C2);
    void* scores = nextWorkspacePtr((int8_t*)ptsData, ptsDataSize);

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
    void* indices = nextWorkspacePtr((int8_t*)scores, scoresSize);

    size_t postNMSScoresSize = detectionForwardPostNMSSize(N, numClasses, topK);
    void* postNMSScores = nextWorkspacePtr((int8_t*)indices, indicesSize);

    size_t postNMSIndicesSize = detectionForwardPostNMSSize(N, numClasses, topK);
    void* postNMSIndices = nextWorkspacePtr((int8_t*)postNMSScores, postNMSScoresSize);

    void* sortingWorkspace = nextWorkspacePtr((int8_t*)postNMSIndices, postNMSIndicesSize);

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

    status = gatherTopDetectionsV2(stream,
                                   shareLocation,
                                   N,
                                   numPredsPerClass,
                                   numClasses,
                                   numPts,
                                   topK,
                                   keepTopK,
                                   DataType::kFLOAT,
                                   DataType::kFLOAT,
                                   indices,
                                   scores,
                                   bboxData,
                                   ptsData,
                                   keepCount,
                                   topDetections);
    if (STATUS_SUCCESS != status) {
        LOG_ERROR("Failed to gatherTopDetectionsV2");
        return status;
    }
    return STATUS_SUCCESS;
}
