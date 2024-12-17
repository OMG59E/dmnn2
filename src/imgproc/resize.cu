/*
 * @Author: xingwg
 * @Date: 2024-10-18 14:22:18
 * @LastEditTime: 2024-12-16 16:27:22
 * @FilePath: /dmnn2/src/imgproc/resize.cu
 * @Description:
 *
 * Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#include "imgproc/resize.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define INC(x, l) ((x + 1) >= (l) ? (x) : ((x) + 1))
#define INTER_RESIZE_COEF_BITS 11
#define INTER_RESIZE_COEF_SCALE (1 << INTER_RESIZE_COEF_BITS)
#define CAST_BITS (INTER_RESIZE_COEF_BITS << 1)
#define CLIP(val, low, high) (val < low ? low : (val > high ? high : val))
static __inline__ __device__ int resize_cast(int value) {
    return (value + (1 << (CAST_BITS - 1))) >> CAST_BITS;
}

inline int getResizeParam(const nv::Image &src, const nv::Image &dst,
                          nv::PaddingMode padding_mode, int &target_h,
                          int &target_w, int &start_y, int &start_x) {
    int input_h = dst.h();
    int input_w = dst.w();
    int src_h = src.h();
    int src_w = src.w();
    float sx = src_w / (float)input_w;
    float sy = src_h / (float)input_h;
    float ratio = src_w / (float)src_h;

    if (padding_mode == nv::PaddingMode::CENTER) {
        if (sx > sy) {
            target_w = input_w;
            target_h = int(target_w / ratio);
            start_y = (input_h - target_h) / 2;
        } else {
            target_h = input_h;
            target_w = int(ratio * target_h);
            start_x = (input_w - target_w) / 2;
        }
    } else if (padding_mode == nv::PaddingMode::TOP_LEFT) {
        if (sx > sy) {
            target_w = input_w;
            target_h = int(target_w / ratio);
        } else {
            target_h = input_h;
            target_w = int(ratio * target_h);
        }
    } else if (padding_mode == nv::PaddingMode::NONE) {
        target_w = input_w;
        target_h = input_h;
    } else {
        LOG_ERROR("unknown padding mode");
        return -1;
    }
    return 0;
}

namespace nv {
template <typename T>
__device__ float calcScaleVal(const T *src, int channels, int src_h, int src_w,
                              int dst_h, int dst_w, int dx, int dy, int dc,
                              nv::ColorType color_type) {
    float scale_x = (float)src_w / dst_w;
    float scale_y = (float)src_h / dst_h;
    float fx = (dx + 0.5f) * scale_x - 0.5f;  // 原图浮点坐标
    float fy = (dy + 0.5f) * scale_y - 0.5f;
    int sx = int(floorf(fx));  // 原图整型坐标
    int sy = int(floorf(fy));  // 原图整型坐标
    fx -= sx;
    fy -= sy;
    // 判断越界
    if (sx < 0) {
        fx = 0.0f;
        sx = 0;
    }
    if (sy < 0) {
        fy = 0.0f;
        sy = 0;
    }
    if (sx >= src_w - 1) {
        fx = 1.0f;
        sx = src_w - 2;
    }
    if (sy >= src_h - 1) {
        fy = 1.0f;
        sy = src_h - 2;
    }
    float2 cbufx, cbufy;
    cbufx.x = 1.0f - fx;
    cbufy.x = 1.0f - fy;
    cbufx.y = fx;
    cbufy.y = fy;
    float v00, v01, v10, v11;
    switch (color_type) {
    case nv::COLOR_TYPE_BGR888_PACKED:
    case nv::COLOR_TYPE_RGB888_PACKED:
        v00 = src[(sy + 0) * src_w * channels + (sx + 0) * channels + dc];
        v01 = src[(sy + 1) * src_w * channels + (sx + 0) * channels + dc];
        v10 = src[(sy + 0) * src_w * channels + (sx + 1) * channels + dc];
        v11 = src[(sy + 1) * src_w * channels + (sx + 1) * channels + dc];
        break;
    case nv::COLOR_TYPE_BGR888_PLANAR:
    case nv::COLOR_TYPE_RGB888_PLANAR:
        v00 = src[dc * src_h * src_w + (sy + 0) * src_w + (sx + 0)];
        v01 = src[dc * src_h * src_w + (sy + 1) * src_w + (sx + 0)];
        v10 = src[dc * src_h * src_w + (sy + 0) * src_w + (sx + 1)];
        v11 = src[dc * src_h * src_w + (sy + 1) * src_w + (sx + 1)];
    default:
        break;
    }
    return cbufx.x * cbufy.x * v00 + cbufx.x * cbufy.y * v01 +
           cbufx.y * cbufy.x * v10 + cbufx.y * cbufy.y * v11;
}

// reference:
// https://github.com/OpenPPL/ppl.cv/blob/04ef4ca48262601b99f1bb918dcd005311f331da/src/ppl/cv/cuda/resize.cu
__device__ unsigned char calcScaleVal2(const unsigned char *src, int channels,
                                       int src_h, int src_w, int dst_h,
                                       int dst_w, int dx, int dy, int dc,
                                       nv::ColorType color_type) {
    float scale_x = (float)src_w / dst_w;
    float scale_y = (float)src_h / dst_h;
    float fx = (dx + 0.5f) * scale_x - 0.5f;  // 原图浮点坐标
    float fy = (dy + 0.5f) * scale_y - 0.5f;
    int sx = floorf(fx);
    int sy = floorf(fy);
    fx -= sx;
    fy -= sy;
    if (sx < 0) {
        sx = 0;
        fx = 0;
    }
    if (sy < 0) {
        sy = 0;
        fy = 0;
    }
    if (sx >= src_w) {
        sx = src_w - 1;
        fx = 0;
    }
    if (sy >= src_h) {
        sy = src_h - 1;
        fy = 0;
    }

    int sx0 = sx;
    int sy0 = sy;
    int sx1 = INC(sx, src_w);
    int sy1 = INC(sy, src_h);
    fx *= INTER_RESIZE_COEF_SCALE;
    fy *= INTER_RESIZE_COEF_SCALE;

    int2 cbufx, cbufy;
    cbufx.x = rint(INTER_RESIZE_COEF_SCALE - rint(fx));
    cbufx.y = rint(fx);
    cbufy.x = rint(INTER_RESIZE_COEF_SCALE - rint(fy));
    cbufy.y = rint(fy);

    float v00, v01, v10, v11;
    switch (color_type) {
    case nv::COLOR_TYPE_BGR888_PACKED:
    case nv::COLOR_TYPE_RGB888_PACKED:  // HWC
        v00 = src[sy0 * src_w * channels + sx0 * channels + dc];
        v01 = src[sy0 * src_w * channels + sx1 * channels + dc];
        v10 = src[sy1 * src_w * channels + sx0 * channels + dc];
        v11 = src[sy1 * src_w * channels + sx1 * channels + dc];
        break;
    case nv::COLOR_TYPE_BGR888_PLANAR:
    case nv::COLOR_TYPE_RGB888_PLANAR:  // CHW
        v00 = src[dc * src_h * src_w + sy0 * src_w + sx0];
        v01 = src[dc * src_h * src_w + sy0 * src_w + sx1];
        v10 = src[dc * src_h * src_w + sy1 * src_w + sx0];
        v11 = src[dc * src_h * src_w + sy1 * src_w + sx1];
    default:
        break;
    }
    return resize_cast(cbufx.x * cbufy.x * v00 + cbufx.y * cbufy.x * v01 +
                       cbufx.x * cbufy.y * v10 + cbufx.y * cbufy.y * v11);
}

__global__ void resizePaddingCvtColorNorm_kernel(
    const int nbThreads, const unsigned char *src, int channels, int src_h,
    int src_w, nv::ColorType src_color_type, float *dst, int dst_h, int dst_w,
    nv::ColorType dst_color_type, int target_h, int target_w, int start_y,
    int start_x, float3 mean_vals, float3 std_vals, float3 padding_values) {
    CUDA_KERNEL_LOOP(idx, nbThreads) {
        int dx = idx % dst_w;
        int dy = idx / dst_w;
        uchar3 pixel;
        if ((dx >= start_x && dx < target_w + start_x) &&
            (dy >= start_y && dy < target_h + start_y)) {
            int tx = dx - start_x;
            int ty = dy - start_y;
            pixel.x = calcScaleVal2(src, channels, src_h, src_w, target_h,
                                    target_w, tx, ty, 0, src_color_type);
            pixel.y = calcScaleVal2(src, channels, src_h, src_w, target_h,
                                    target_w, tx, ty, 1, src_color_type);
            pixel.z = calcScaleVal2(src, channels, src_h, src_w, target_h,
                                    target_w, tx, ty, 2, src_color_type);
        } else {
            pixel.x = padding_values.x;
            pixel.y = padding_values.y;
            pixel.z = padding_values.z;
        }

        switch (dst_color_type) {
        case nv::COLOR_TYPE_BGR888_PACKED:
            dst[dy * dst_w * channels + dx * channels + 0] =
                (pixel.x - mean_vals.x) / std_vals.x;
            dst[dy * dst_w * channels + dx * channels + 1] =
                (pixel.y - mean_vals.y) / std_vals.y;
            dst[dy * dst_w * channels + dx * channels + 2] =
                (pixel.z - mean_vals.z) / std_vals.z;
            break;
        case nv::COLOR_TYPE_RGB888_PACKED:
            dst[dy * dst_w * channels + dx * channels + 2] =
                (pixel.x - mean_vals.x) / std_vals.x;
            dst[dy * dst_w * channels + dx * channels + 1] =
                (pixel.y - mean_vals.y) / std_vals.y;
            dst[dy * dst_w * channels + dx * channels + 0] =
                (pixel.z - mean_vals.z) / std_vals.z;
            break;
        case nv::COLOR_TYPE_BGR888_PLANAR:
            dst[0 * dst_h * dst_w + dy * dst_w + dx] =
                (pixel.x - mean_vals.x) / std_vals.x;
            dst[1 * dst_h * dst_w + dy * dst_w + dx] =
                (pixel.y - mean_vals.y) / std_vals.y;
            dst[2 * dst_h * dst_w + dy * dst_w + dx] =
                (pixel.z - mean_vals.z) / std_vals.z;
            break;
        case nv::COLOR_TYPE_RGB888_PLANAR:
            dst[2 * dst_h * dst_w + dy * dst_w + dx] =
                (pixel.x - mean_vals.x) / std_vals.x;
            dst[1 * dst_h * dst_w + dy * dst_w + dx] =
                (pixel.y - mean_vals.y) / std_vals.y;
            dst[0 * dst_h * dst_w + dy * dst_w + dx] =
                (pixel.z - mean_vals.z) / std_vals.z;
            break;
        default:
            printf("Not support color convert\n");
            return;
        }
    }
}

__global__ void resizePaddingCvtColor_kernel(
    const int nbThreads, const unsigned char *src, int channels, int src_h,
    int src_w, nv::ColorType src_color_type, unsigned char *dst, int dst_h,
    int dst_w, nv::ColorType dst_color_type, int target_h, int target_w,
    int start_y, int start_x, float3 padding_values) {
    CUDA_KERNEL_LOOP(idx, nbThreads) {
        int dx = idx % dst_w;
        int dy = idx / dst_w;
        uchar3 pixel;
        if ((dx >= start_x && dx < target_w + start_x) &&
            (dy >= start_y && dy < target_h + start_y)) {
            int tx = dx - start_x;
            int ty = dy - start_y;
            pixel.x = calcScaleVal2(src, channels, src_h, src_w, target_h,
                                    target_w, tx, ty, 0, src_color_type);
            pixel.y = calcScaleVal2(src, channels, src_h, src_w, target_h,
                                    target_w, tx, ty, 1, src_color_type);
            pixel.z = calcScaleVal2(src, channels, src_h, src_w, target_h,
                                    target_w, tx, ty, 2, src_color_type);
        } else {
            pixel.x = padding_values.x;
            pixel.y = padding_values.y;
            pixel.z = padding_values.z;
        }

        switch (dst_color_type) {
        case nv::COLOR_TYPE_BGR888_PACKED:
            dst[dy * dst_w * channels + dx * channels + 0] = pixel.x;
            dst[dy * dst_w * channels + dx * channels + 1] = pixel.y;
            dst[dy * dst_w * channels + dx * channels + 2] = pixel.z;
            break;
        case nv::COLOR_TYPE_RGB888_PACKED:
            dst[dy * dst_w * channels + dx * channels + 2] = pixel.x;
            dst[dy * dst_w * channels + dx * channels + 1] = pixel.y;
            dst[dy * dst_w * channels + dx * channels + 0] = pixel.z;
            break;
        case nv::COLOR_TYPE_BGR888_PLANAR:
            dst[0 * dst_h * dst_w + dy * dst_w + dx] = pixel.x;
            dst[1 * dst_h * dst_w + dy * dst_w + dx] = pixel.y;
            dst[2 * dst_h * dst_w + dy * dst_w + dx] = pixel.z;
            break;
        case nv::COLOR_TYPE_RGB888_PLANAR:
            dst[2 * dst_h * dst_w + dy * dst_w + dx] = pixel.x;
            dst[1 * dst_h * dst_w + dy * dst_w + dx] = pixel.y;
            dst[0 * dst_h * dst_w + dy * dst_w + dx] = pixel.z;
            break;
        default:
            printf("Not support color convert\n");
            return;
        }
    }
}

__global__ void resize_kernel(const int nbThreads, const unsigned char *src,
                              int channels, int src_h, int src_w,
                              nv::ColorType src_color_type, unsigned char *dst,
                              int dst_h, int dst_w,
                              nv::ColorType dst_color_type) {
    CUDA_KERNEL_LOOP(idx, nbThreads) {
        int dx = idx % dst_w;
        int dy = idx / dst_w;
        uchar3 pixel;
        pixel.x = calcScaleVal2(src, channels, src_h, src_w, dst_h, dst_w, dx,
                                dy, 0, src_color_type);
        pixel.y = calcScaleVal2(src, channels, src_h, src_w, dst_h, dst_w, dx,
                                dy, 1, src_color_type);
        pixel.z = calcScaleVal2(src, channels, src_h, src_w, dst_h, dst_w, dx,
                                dy, 2, src_color_type);

        switch (dst_color_type) {
        case nv::COLOR_TYPE_RGB888_PLANAR:
        case nv::COLOR_TYPE_BGR888_PLANAR:
            dst[0 * dst_h * dst_w + dy * dst_w + dx] = pixel.x;
            dst[1 * dst_h * dst_w + dy * dst_w + dx] = pixel.y;
            dst[2 * dst_h * dst_w + dy * dst_w + dx] = pixel.z;
            break;
        case nv::COLOR_TYPE_RGB888_PACKED:
        case nv::COLOR_TYPE_BGR888_PACKED:
            dst[dy * dst_w * channels + dx * channels + 0] = pixel.x;
            dst[dy * dst_w * channels + dx * channels + 1] = pixel.y;
            dst[dy * dst_w * channels + dx * channels + 2] = pixel.z;
            break;
        default:
            printf("Not support color convert\n");
            break;
        }
    }
}

int resizePaddingCvtColorNormAsync(cudaStream_t stream, const nv::Image &src,
                                   nv::Image &dst, float *mean_vals,
                                   float *std_vals, PaddingMode padding_mode,
                                   float *padding_values) {
    LOG_ASSERT(dst.dataType == nv::DataType::DATA_TYPE_FLOAT32 &&
               src.dataType == nv::DataType::DATA_TYPE_UINT8);
    int target_h = 0, target_w = 0;
    int start_x = 0, start_y = 0;
    if (0 != getResizeParam(src, dst, padding_mode, target_h, target_w, start_y,
                            start_x)) {
        LOG_ERROR("getResizeParam failed");
        return -1;
    }
    float3 _mean_vals = make_float3(mean_vals[0], mean_vals[1], mean_vals[2]);
    float3 _std_vals = make_float3(std_vals[0], std_vals[1], std_vals[2]);
    float3 _padding_values =
        make_float3(padding_values[0], padding_values[1], padding_values[2]);
    const int nbThreads = dst.h() * dst.w();
    resizePaddingCvtColorNorm_kernel<<<CUDA_GET_BLOCKS(nbThreads),
                                       CUDA_NUM_THREADS, 0, stream>>>(
        nbThreads, (unsigned char *)(src.gpu_data), src.channels(), src.h(),
        src.w(), src.colorType, (float *)(dst.gpu_data), dst.h(), dst.w(),
        dst.colorType, target_h, target_w, start_y, start_x, _mean_vals,
        _std_vals, _padding_values);
    CUDACHECK(cudaStreamSynchronize(stream));
    CUDACHECK(cudaGetLastError());
    return 0;
}

int resizePaddingCvtColorNorm(const nv::Image &src, nv::Image &dst,
                              float *mean_vals, float *std_vals,
                              PaddingMode padding_mode, float *padding_values) {
    LOG_ASSERT(dst.dataType == nv::DataType::DATA_TYPE_FLOAT32 &&
               src.dataType == nv::DataType::DATA_TYPE_UINT8);
    int target_h = 0, target_w = 0;
    int start_x = 0, start_y = 0;
    if (0 != getResizeParam(src, dst, padding_mode, target_h, target_w, start_y,
                            start_x)) {
        LOG_ERROR("getResizeParam failed");
        return -1;
    }
    float3 _mean_vals = make_float3(mean_vals[0], mean_vals[1], mean_vals[2]);
    float3 _std_vals = make_float3(std_vals[0], std_vals[1], std_vals[2]);
    float3 _padding_values =
        make_float3(padding_values[0], padding_values[1], padding_values[2]);
    const int nbThreads = dst.h() * dst.w();
    resizePaddingCvtColorNorm_kernel<<<CUDA_GET_BLOCKS(nbThreads),
                                       CUDA_NUM_THREADS>>>(
        nbThreads, (unsigned char *)(src.gpu_data), src.channels(), src.h(),
        src.w(), src.colorType, (float *)(dst.gpu_data), dst.h(), dst.w(),
        dst.colorType, target_h, target_w, start_y, start_x, _mean_vals,
        _std_vals, _padding_values);
    CUDACHECK(cudaDeviceSynchronize());
    CUDACHECK(cudaGetLastError());
    return 0;
}

int resizePaddingCvtColorAsync(cudaStream_t stream, const nv::Image &src,
                               nv::Image &dst, PaddingMode padding_mode,
                               float *padding_values) {
    LOG_ASSERT(dst.dataType == nv::DataType::DATA_TYPE_UINT8 &&
               src.dataType == nv::DataType::DATA_TYPE_UINT8);
    int target_h = 0, target_w = 0;
    int start_x = 0, start_y = 0;
    if (0 != getResizeParam(src, dst, padding_mode, target_h, target_w, start_y,
                            start_x)) {
        LOG_ERROR("getResizeParam failed");
        return -1;
    }
    float3 _padding_values =
        make_float3(padding_values[0], padding_values[1], padding_values[2]);
    const int nbThreads = dst.h() * dst.h();
    resizePaddingCvtColor_kernel<<<CUDA_GET_BLOCKS(nbThreads), CUDA_NUM_THREADS,
                                   0, stream>>>(
        nbThreads, (unsigned char *)(src.gpu_data), src.channels(), src.h(),
        src.w(), src.colorType, (unsigned char *)(dst.gpu_data), dst.h(),
        dst.w(), dst.colorType, target_h, target_w, start_y, start_x,
        _padding_values);
    CUDACHECK(cudaStreamSynchronize(stream));
    CUDACHECK(cudaGetLastError());
    return 0;
}

int resizePaddingCvtColor(const nv::Image &src, nv::Image &dst,
                          PaddingMode padding_mode, float *padding_values) {
    LOG_ASSERT(dst.dataType == nv::DataType::DATA_TYPE_UINT8 &&
               src.dataType == nv::DataType::DATA_TYPE_UINT8);
    int target_h = 0, target_w = 0;
    int start_x = 0, start_y = 0;
    if (0 != getResizeParam(src, dst, padding_mode, target_h, target_w, start_y,
                            start_x)) {
        LOG_ERROR("getResizeParam failed");
        return -1;
    }
    float3 _padding_values =
        make_float3(padding_values[0], padding_values[1], padding_values[2]);
    const int nbThreads = dst.h() * dst.h();
    resizePaddingCvtColor_kernel<<<CUDA_GET_BLOCKS(nbThreads),
                                   CUDA_NUM_THREADS>>>(
        nbThreads, (unsigned char *)(src.gpu_data), src.channels(), src.h(),
        src.w(), src.colorType, (unsigned char *)(dst.gpu_data), dst.h(),
        dst.w(), dst.colorType, target_h, target_w, start_y, start_x,
        _padding_values);
    CUDACHECK(cudaDeviceSynchronize());
    CUDACHECK(cudaGetLastError());
    return 0;
}

int resize(const nv::Image &src, nv::Image &dst) {
    LOG_ASSERT(src.gpu_data && dst.gpu_data);
    LOG_ASSERT(dst.dataType == nv::DataType::DATA_TYPE_UINT8 &&
               src.dataType == nv::DataType::DATA_TYPE_UINT8);
    LOG_ASSERT(src.colorType == dst.colorType);
    const int nbThreads = dst.w() * dst.h();
    resize_kernel<<<CUDA_GET_BLOCKS(nbThreads), CUDA_NUM_THREADS>>>(
        nbThreads, (unsigned char *)(src.gpu_data), src.channels(), src.h(),
        src.w(), src.colorType, (unsigned char *)(dst.gpu_data), dst.h(),
        dst.w(), dst.colorType);
    CUDACHECK(cudaDeviceSynchronize());
    CUDACHECK(cudaGetLastError());
    return 0;
}
}  // namespace nv