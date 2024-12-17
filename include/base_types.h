/***
 * @Author: xingwg
 * @Date: 2024-10-11 16:03:10
 * @LastEditTime: 2024-11-21 11:48:38
 * @FilePath: /dmnn2/include/base_types.h
 * @Description:
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#pragma once
#include "error_check.h"
#include "logging.h"
#include <random>

#ifdef AARCH64
// CUDA: use 512 threads per block
#define CUDA_NUM_THREADS 64
#else
#define CUDA_NUM_THREADS 512
#endif

#define CUDA_GET_BLOCKS(N) (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS
#define CUDA_KERNEL_LOOP(i, n)                                                 \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);               \
         i += blockDim.x * gridDim.x)

#define FLOAT_MAX 3.402823466e+38F /* max value */

namespace nv {
struct DimsCHW {
    int nbDims{3};
    int d[8]{};
    DimsCHW() = default;
    DimsCHW(int c, int h, int w) {
        d[0] = c;
        d[1] = h;
        d[2] = w;
    }
    int c() const { return d[0]; }
    int h() const { return d[1]; }
    int w() const { return d[2]; }
    int size() const { return d[0] * d[1] * d[2]; }
};

struct DimsNCHW {
    int nbDims{4};
    int d[8]{};
    DimsNCHW() = default;
    DimsNCHW(int n, int c, int h, int w) {
        d[0] = n;
        d[1] = c;
        d[2] = h;
        d[3] = w;
    }
    int n() const { return d[0]; }
    int c() const { return d[1]; }
    int h() const { return d[2]; }
    int w() const { return d[3]; }
    int size() const { return d[0] * d[1] * d[2] * d[3]; }
};

typedef enum ColorType {
    COLOR_TYPE_UNKNOWN = -1,
    COLOR_TYPE_GRAY = 0,
    COLOR_TYPE_BGR888_PACKED,    // BGRBGRBGRBGR...
    COLOR_TYPE_BGR888_PLANAR,    // BBBB...GGGG...RRRR...
    COLOR_TYPE_RGB888_PACKED,    // RGBRGBRGBRGB...
    COLOR_TYPE_RGB888_PLANAR,    // RRRR...GGGG...BBBB...
    COLOR_TYPE_BGRA8888_PACKED,  // BGRABGRABGRABGRA...
    COLOR_TYPE_RGBA8888_PACKED,  // RGBARGBARGBARGBA...
    COLOR_TYPE_YUV420SP_NV12,    // YYYYYYYYYYYYYYYY UVUVUVUV
    COLOR_TYPE_YUV420SP_NV21,    // YYYYYYYYYYYYYYYY VUVUVUVU
    COLOR_TYPE_YUV420P_YU12,     // YYYYYYYYYYYYYYYY UUUUVVVV
    COLOR_TYPE_YUV420P_YV12,     // YYYYYYYYYYYYYYYY VVVVUUUU
    COLOR_TYPE_YUV422P,
    COLOR_TYPE_YUV444P,
} colorType_t;

typedef enum ColorSpace {
    COLOR_UNKNOWN = -1,
    COLOR_TYPE_BGR888_PACKED_TO_RGB888_PACKED = 0,
    COLOR_TYPE_BGR888_PACKED_TO_RGB888_PLANAR,
} colorSpace_t;

typedef enum DataType {
    DATA_TYPE_UNKNOWN = -1,
    DATA_TYPE_FLOAT32 = 0,
    DATA_TYPE_INT8,
    DATA_TYPE_UINT8,
    DATA_TYPE_INT16,
    DATA_TYPE_UINT16,
    DATA_TYPE_INT32,
    DATA_TYPE_UINT32,
    DATA_TYPE_INT64,
    DATA_TYPE_UINT64,
    DATA_TYPE_BOOL,
    DATA_TYPE_FLOAT16,
    DATA_TYPE_DOUBLE
} dataType_t;

typedef enum MemoryType { MEMORY_TYPE_CPU = 0, MEMORY_TYPE_GPU } memoryType_t;

static uint8_t get_item_size(DataType dataType) {
    switch (dataType) {
    case DATA_TYPE_BOOL:
    case DATA_TYPE_INT8:
    case DATA_TYPE_UINT8:
        return 1;
    case DATA_TYPE_FLOAT16:
    case DATA_TYPE_INT16:
    case DATA_TYPE_UINT16:
        return 2;
    case DATA_TYPE_FLOAT32:
    case DATA_TYPE_INT32:
    case DATA_TYPE_UINT32:
        return 4;
    case DATA_TYPE_INT64:
    case DATA_TYPE_UINT64:
    case DATA_TYPE_DOUBLE:
        return 8;
    default:
        LOG_FATAL("Unknown data type: {}", static_cast<int>(dataType));
    }
}

typedef struct Image {
    int32_t width{0};
    int32_t height{0};
    void *data{nullptr};
    void *gpu_data{nullptr};
    bool own{true};
    DataType dataType{DATA_TYPE_UINT8};
    ColorType colorType{COLOR_TYPE_BGR888_PACKED};
    int channels() const {
        if (colorType == COLOR_TYPE_GRAY) {
            return 1;
        } else if (colorType == COLOR_TYPE_RGB888_PACKED ||
                   colorType == COLOR_TYPE_BGR888_PACKED ||
                   colorType == COLOR_TYPE_BGR888_PLANAR ||
                   colorType == COLOR_TYPE_RGB888_PLANAR ||
                   colorType == COLOR_TYPE_YUV420SP_NV12 ||
                   colorType == COLOR_TYPE_YUV420SP_NV21 ||
                   colorType == COLOR_TYPE_YUV420P_YV12 ||
                   colorType == COLOR_TYPE_YUV420P_YU12 ||
                   colorType == COLOR_TYPE_YUV422P ||
                   colorType == COLOR_TYPE_YUV444P) {
            return 3;
        } else if (colorType == COLOR_TYPE_RGBA8888_PACKED ||
                   colorType == COLOR_TYPE_BGRA8888_PACKED) {
            return 4;
        } else {
            LOG_FATAL("unknown color type");
        }
    }
    int h() const { return height; }
    int w() const { return width; }
    int size() const {
        if (colorType == COLOR_TYPE_GRAY) {
            return width * height;
        } else if (colorType == COLOR_TYPE_RGB888_PACKED ||
                   colorType == COLOR_TYPE_BGR888_PACKED ||
                   colorType == COLOR_TYPE_BGR888_PLANAR ||
                   colorType == COLOR_TYPE_RGB888_PLANAR ||
                   colorType == COLOR_TYPE_YUV444P) {
            return width * height * 3;
        } else if (colorType == COLOR_TYPE_YUV420SP_NV12 ||
                   colorType == COLOR_TYPE_YUV420SP_NV21 ||
                   colorType == COLOR_TYPE_YUV420P_YV12 ||
                   colorType == COLOR_TYPE_YUV420P_YU12) {
            return width * height * 3 / 2;
        } else if (colorType == COLOR_TYPE_YUV422P) {
            return width * height * 2;
        } else if (colorType == COLOR_TYPE_RGBA8888_PACKED ||
                   colorType == COLOR_TYPE_BGRA8888_PACKED) {
            return width * height * 4;
        } else {
            LOG_FATAL("unknown color type");
        }
    }
    uint8_t item_size() const { return get_item_size(dataType); }
    int size_bytes() const { return size() * item_size(); }
    void free() {
        if (own) {
            CUDA_FREE(gpu_data);
            CUDA_HOST_FREE(data);
        }
    }
    void create(int h, int w, bool gpu = true,
                ColorType _colorType = COLOR_TYPE_BGR888_PACKED,
                DataType _dataType = DATA_TYPE_UINT8) {
        width = w;
        height = h;
        dataType = _dataType;
        colorType = _colorType;
        own = true;
        LOG_ASSERT(!data && !gpu_data);
        if (gpu) {
            CUDACHECK(cudaMalloc(&gpu_data, size_bytes()));
        } else {
            CUDACHECK(cudaMallocHost(&data, size_bytes()));
        }
    }
} image_t;

#define TESNOR_MAX_DIM 8
struct Tensor {
    int32_t idx;
    std::string name;
    uint32_t nbDims{TESNOR_MAX_DIM};
    uint32_t dims[TESNOR_MAX_DIM]{};
    DataType dataType{DATA_TYPE_FLOAT32};
    bool own{true};
    void *data{nullptr};
    void *gpu_data{nullptr};
    size_t size() const {
        size_t v = 1;
        for (int d = 0; d < nbDims; ++d)
            v *= dims[d];
        return v;
    }
    uint8_t item_size() const { return get_item_size(dataType); }
    size_t size_bytes() const { return size() * item_size(); }
    void free() {
        if (own) {
            CUDA_FREE(gpu_data);
            CUDA_HOST_FREE(data);
        }
    }
};

typedef struct Color {
    int r{0};
    int g{0};
    int b{0};
    Color() {
        // 使用 random_device 来获得种子
        std::random_device rd;
        std::mt19937 gen(rd());  // Mersenne Twister 引擎
        std::uniform_int_distribution<> dis(
            0, 255);   // 均匀分布，生成 0 到 255 之间的整数
        r = dis(gen);  // 随机生成 r 值
        g = dis(gen);  // 随机生成 g 值
        b = dis(gen);  // 随机生成 b 值
    }
    Color(int _r, int _g, int _b) {
        r = _r;
        g = _g;
        b = _b;
    }
} color_t;

typedef struct Point {
    int x{0};
    int y{0};
    Point() = default;
    Point(int _x, int _y) {
        x = _x;
        y = _y;
    }
} point_t;

struct Size {
    int w{0};
    int h{0};
};

typedef struct BoundingBox {
    int x1{0};
    int y1{0};
    int x2{0};
    int y2{0};
    int w() const { return x2 - x1 + 1; }
    int h() const { return y2 - y1 + 1; }
    int area() const { return w() * h(); }
    int cx() const { return (x1 + x2) / 2; }
    int cy() const { return (y1 + y2) / 2; }
    BoundingBox() = default;
    BoundingBox(int x1, int x2, int y1, int y2) {
        this->x1 = x1;
        this->x2 = x2;
        this->y1 = y1;
        this->y2 = y2;
    }
    color_t color;
} bbox_t;

typedef struct Classification {
    float score{0};
    int32_t cls_idx{0};
} classification_t;

typedef struct Detetcion {
    BoundingBox bbox;
    float score{0};
    int32_t cls_idx{0};
} detection_t;

typedef enum PaddingMode {
    NONE = -1,
    CENTER = 0,
    TOP_LEFT,
} padding_mode_t;

typedef std::vector<detection_t> detections_t;
typedef std::vector<classification_t> classifications_t;

// 目标定义
typedef struct Object {
    BoundingBox bbox;                 // 当前帧目标框
    int32_t cls_idx{0};               // 目标类别
    float det_score{0};               // 检测分数
    float cls_score{0};               // 分类分数
    std::vector<BoundingBox> bboxes;  // 目标轨迹
} object_t;

typedef struct Frame : Image {
    uint64_t channel_id{0};  // 通道号
    uint64_t idx{0};         // 帧号
    uint64_t timestamp{
        0};  // 流：被解码时刻的时间戳，文件：依据帧率换算的时间戳, 微秒
    std::vector<object_t> objects;  // 当前帧目标信息
} frame_t;
}  // namespace nv