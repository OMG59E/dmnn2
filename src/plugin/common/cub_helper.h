/*** 
 * @Author: xingwg
 * @Date: 2024-10-12 09:59:37
 * @LastEditTime: 2024-10-12 10:54:17
 * @FilePath: /dmnn2/src/plugin/common/cub_helper.h
 * @Description: 
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved. 
 */

#include "kernels/kernel.h"
#include <cub/cub.cuh>

template<typename KeyT, typename ValueT>
size_t cubSortPairsWorkspaceSize(int num_items, int num_segments) {
    size_t temp_storage_bytes = 0;
    cub::DeviceSegmentedRadixSort::SortPairsDescending(
        (void *) NULL, temp_storage_bytes, (const KeyT *) NULL, (KeyT *) NULL, (const ValueT *) NULL, (ValueT *) NULL, num_items,  num_segments, (const int *) NULL, (const int *) NULL);
    return temp_storage_bytes;
}
