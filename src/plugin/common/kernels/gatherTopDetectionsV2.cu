/*
 * @Author: xingwg
 * @Date: 2024-10-12 09:59:37
 * @LastEditTime: 2024-10-12 10:34:59
 * @FilePath: /dmnn2/src/plugin/common/kernels/gatherTopDetectionsV2.cu
 * @Description: 
 * 
 * Copyright (c) 2024 by Chinasvt, All Rights Reserved. 
 */
#include <vector>
#include "../plugin.h"
#include "kernel.h"

template<typename T_BBOX, typename T_SCORE, unsigned nthds_per_cta>
__launch_bounds__(nthds_per_cta)
__global__ void gatherTopDetectionsV2_kernel(
        const bool shareLocation,
        const int numImages,
        const int numPredsPerClass,
        const int numClasses,
        const int numPts,
        const int topK,
        const int keepTopK,
        const int *indices,
        const T_SCORE *scores,
        const T_BBOX *bboxData,
        const T_BBOX *ptsData,
        int *keepCount,
        T_BBOX *topDetections) {
    if (keepTopK > topK)
        return;
    for (int i = blockIdx.x * nthds_per_cta + threadIdx.x; i < numImages * keepTopK; i += gridDim.x * nthds_per_cta) {
        const int imgId = i / keepTopK;
        const int detId = i % keepTopK;
        const int offset = imgId * numClasses * topK;
        const int index = indices[offset + detId];
        const T_SCORE score = scores[offset + detId];
        /*
         * It is also likely that there is "bad bounding boxes" in the keepTopK bounding boxes.
         * We set the bounding boxes parameters as the parameters shown below.
         * These data will only show up at the end of keepTopK bounding boxes since the bounding boxes were sorted previously.
         * It is also not going to affect the count of valid bounding boxes (keepCount).
         * These data will probably never be used (because we have keepCount).
         */
        const int step = 7 + (2 * numPts);
        if (index == -1) {
            for (int k = 0; k < step; k++) {
                if (k == 0) {
                    topDetections[i * step + k] = imgId;  // image id
                } else if (k == 1) {
                    topDetections[i * step + k] = -1; // label
                } else {
                    topDetections[i * step + k] = 0;
                }
            }
        } else {
            const int bboxOffset = imgId * (shareLocation ? numPredsPerClass : (numClasses * numPredsPerClass));
            const int bboxId = ((shareLocation ? (index % numPredsPerClass)
                                               : index % (numClasses * numPredsPerClass)) + bboxOffset) * 4;

            const int ptsId = ((shareLocation ? (index % numPredsPerClass)
                                              : index % (numClasses * numPredsPerClass)) + bboxOffset) * (2 * numPts);

            for (int k = 0; k < step; k++) {
                if (k == 0) {
                    topDetections[i * step + k] = imgId;  // image id
                } else if (k == 1) {
                    topDetections[i * step + k] = (index % (numClasses * numPredsPerClass)) / numPredsPerClass; // label
                } else if (k == 2) {
                    topDetections[i * step + k] = score;
                } else if (k <= 6) {
                    topDetections[i * step + k] = bboxData[bboxId + k - 3];
                } else {
                    topDetections[i * step + k] = ptsData[ptsId + k - 7];  //max(min(ptsData[ptsId + k - 7], T_BBOX(1.)), T_BBOX(0.));
                }
            }

            // Atomic add to increase the count of valid keepTopK bounding boxes
            // Without having to do manual sync.
            //atomicAdd(&keepCount[i / keepTopK], 1);
        }
    }
}

template<typename T_BBOX, typename T_SCORE>
pluginStatus_t gatherTopDetectionsV2_gpu(
        cudaStream_t stream,
        const bool shareLocation,
        const int numImages,
        const int numPredsPerClass,
        const int numClasses,
        const int numPts,
        const int topK,
        const int keepTopK,
        const void *indices,
        const void *scores,
        const void *bboxData,
        const void *ptsData,
        void *keepCount,
        void *topLKDetections) {
    //cudaMemsetAsync(keepCount, 0, numImages * sizeof(int), stream);
    const int BS = 32;
    const int GS = 32;
    gatherTopDetectionsV2_kernel<T_BBOX, T_SCORE, BS><<<GS, BS, 0, stream>>>(shareLocation, numImages, numPredsPerClass,
                                                                             numClasses, numPts, topK, keepTopK,
                                                                             (int *) indices, (T_SCORE *) scores,
                                                                             (T_BBOX *) bboxData,
                                                                             (T_BBOX *) ptsData, (int *) keepCount,
                                                                             (T_BBOX *) topLKDetections);

    CUDACHECK(cudaGetLastError());
    return STATUS_SUCCESS;
}

// gatherTopDetections LAUNCH CONFIG
typedef pluginStatus_t (*gtdLKFunc)(cudaStream_t,
                                    const bool,
                                    const int,
                                    const int,
                                    const int,
                                    const int,
                                    const int,
                                    const int,
                                    const void *,
                                    const void *,
                                    const void *,
                                    const void *,
                                    void *,
                                    void *);

struct gtdV2LaunchConfig {
    DataType t_bbox;
    DataType t_score;
    gtdLKFunc function;

    gtdV2LaunchConfig(DataType t_bbox, DataType t_score)
            : t_bbox(t_bbox), t_score(t_score) {
    }

    gtdV2LaunchConfig(DataType t_bbox, DataType t_score, gtdLKFunc function)
            : t_bbox(t_bbox), t_score(t_score), function(function) {
    }

    bool operator==(const gtdV2LaunchConfig &other) {
        return t_bbox == other.t_bbox && t_score == other.t_score;
    }
};

using nvinfer1::DataType;

static std::vector<gtdV2LaunchConfig> gtdV2FuncVec;

bool gtdV2Init() {
    gtdV2FuncVec.push_back(gtdV2LaunchConfig(DataType::kFLOAT, DataType::kFLOAT,
                                             gatherTopDetectionsV2_gpu<float, float>));
    return true;
}

static bool initialized = gtdV2Init();

pluginStatus_t gatherTopDetectionsV2(
        cudaStream_t stream,
        bool shareLocation,
        int numImages,
        int numPredsPerClass,
        int numClasses,
        int numPts,
        int topK,
        int keepTopK,
        DataType DT_BBOX,
        DataType DT_SCORE,
        const void *indices,
        const void *scores,
        const void *bboxData,
        const void *ptsData,
        void *keepCount,
        void *topLKDetections) {
    gtdV2LaunchConfig lc = gtdV2LaunchConfig(DT_BBOX, DT_SCORE);
    for (unsigned i = 0; i < gtdV2FuncVec.size(); ++i) {
        if (lc == gtdV2FuncVec[i]) {
            return gtdV2FuncVec[i].function(stream,
                                            shareLocation,
                                            numImages,
                                            numPredsPerClass,
                                            numClasses,
                                            numPts,
                                            topK,
                                            keepTopK,
                                            indices,
                                            scores,
                                            bboxData,
                                            ptsData,
                                            keepCount,
                                            topLKDetections);
        }
    }
    return STATUS_BAD_PARAM;
}
