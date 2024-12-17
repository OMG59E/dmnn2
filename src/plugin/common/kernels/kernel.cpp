/*** 
 * @Author: xingwg
 * @Date: 2024-10-12 09:59:37
 * @LastEditTime: 2024-10-12 10:35:18
 * @FilePath: /dmnn2/src/plugin/common/kernels/kernel.cpp
 * @Description: 
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved. 
 */
#include "kernel.h"
#include "../plugin.h"
#include "error_check.h"

size_t detectionInferenceWorkspaceSize(bool shareLocation, int N, int C1, int C2, int numClasses, int numPredsPerClass,
                                       int topK, DataType DT_BBOX, DataType DT_SCORE) {
    size_t wss[7];
    wss[0] = detectionForwardBBoxDataSize(N, C1, DT_BBOX);
    wss[1] = detectionForwardBBoxPermuteSize(shareLocation, N, C1, DT_BBOX);
    wss[2] = detectionForwardPreNMSSize(N, C2);
    wss[3] = detectionForwardPreNMSSize(N, C2);
    wss[4] = detectionForwardPostNMSSize(N, numClasses, topK);
    wss[5] = detectionForwardPostNMSSize(N, numClasses, topK);
    wss[6] = std::max(sortScoresPerClassWorkspaceSize(N, numClasses, numPredsPerClass, DT_SCORE), sortScoresPerImageWorkspaceSize(N, numClasses * topK, DT_SCORE));
    return calculateTotalWorkspaceSize(wss, 7);
}


size_t detectionInferenceWorkspaceSize(int N, int C1, int C2, int numClasses,
                                       int topK, DataType DT_BBOX, DataType DT_SCORE) {
    size_t wss[7];
    wss[0] = detectionForwardBBoxDataSize(N, C1 * 4, DT_BBOX);
    wss[1] = detectionForwardConfDataSize(N, C1 * numClasses, DT_SCORE);
    wss[2] = detectionForwardPreNMSSize(N, C1 * numClasses);
    wss[3] = detectionForwardPreNMSSize(N, C1 * numClasses);
    wss[4] = detectionForwardPostNMSSize(N, numClasses, topK);
    wss[5] = detectionForwardPostNMSSize(N, numClasses, topK);
    wss[6] = std::max(sortScoresPerClassWorkspaceSize(N, numClasses, C1, DT_SCORE), sortScoresPerImageWorkspaceSize(N, numClasses * topK, DT_SCORE));
    return calculateTotalWorkspaceSize(wss, 7);
}

size_t detectionInferenceWorkspaceSize(int N, int C1, int C2, int numClasses, int numPts,
                                       int topK, DataType DT_BBOX, DataType DT_SCORE) {
    size_t wss[8];
    wss[0] = detectionForwardBBoxDataSize(N, C1 * 4, DT_BBOX);
    wss[1] = detectionForwardBBoxDataSize(N, C1 * 2 * numPts, DT_BBOX);
    wss[2] = detectionForwardConfDataSize(N, C1 * numClasses, DT_SCORE);
    wss[3] = detectionForwardPreNMSSize(N, C1 * numClasses);
    wss[4] = detectionForwardPreNMSSize(N, C1 * numClasses);
    wss[5] = detectionForwardPostNMSSize(N, numClasses, topK);
    wss[6] = detectionForwardPostNMSSize(N, numClasses, topK);
    wss[7] = std::max(sortScoresPerClassWorkspaceSize(N, numClasses, C1, DT_SCORE), sortScoresPerImageWorkspaceSize(N, numClasses * topK, DT_SCORE));
    return calculateTotalWorkspaceSize(wss, 7);
}

size_t detectionInferenceWorkspaceSize(bool shareLocation, int N, int C1, int C2, int C3,
        int numClasses, int numPredsPerClass, int numPts, int topK, DataType DT_BBOX,
        DataType DT_SCORE, DataType DT_PTS) {
    size_t wss[8];
    wss[0] = detectionForwardBBoxDataSize(N, C1, DT_BBOX);
    wss[1] = detectionForwardBBoxPermuteSize(shareLocation, N, C1, DT_BBOX);
    wss[2] = detectionForwardPtsDataSize(N, C3, DT_PTS);
    //wss[3] = detectionForwardPtsPermuteSize(shareLocation, N, C3, DT_PTS);
    wss[3] = detectionForwardPreNMSSize(N, C2);
    wss[4] = detectionForwardPreNMSSize(N, C2);
    wss[5] = detectionForwardPostNMSSize(N, numClasses, topK);
    wss[6] = detectionForwardPostNMSSize(N, numClasses, topK);
    wss[7] = std::max(sortScoresPerClassWorkspaceSize(N, numClasses, numPredsPerClass, DT_SCORE), sortScoresPerImageWorkspaceSize(N, numClasses * topK, DT_SCORE));
    return calculateTotalWorkspaceSize(wss, 8);
}
