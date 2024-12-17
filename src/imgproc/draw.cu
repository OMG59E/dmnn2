/*
 * @Author: xingwg
 * @Date: 2024-10-22 15:14:24
 * @LastEditTime: 2024-10-23 11:19:43
 * @FilePath: /dmnn2/src/imgproc/draw.cu
 * @Description:
 *
 * Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#include "imgproc/draw.h"

#define CLIP(a, b) (a < b ? a : b)

namespace nv {
__global__ void line_hv_kernel(const int nbThreads, unsigned char *src,
                               int channels, int src_h, int src_w,
                               nv::ColorType src_color_type, int line_h,
                               int line_w, int x1, int y1, nv::Color color) {
    CUDA_KERNEL_LOOP(idx, nbThreads) {
        int dx = idx % line_w;
        int dy = idx / line_w;

        switch (src_color_type) {
        case nv::COLOR_TYPE_BGR888_PACKED:
            src[(dy + y1) * src_w * channels + (dx + x1) * channels + 0] =
                color.b;
            src[(dy + y1) * src_w * channels + (dx + x1) * channels + 1] =
                color.g;
            src[(dy + y1) * src_w * channels + (dx + x1) * channels + 2] =
                color.r;
            break;
        case nv::COLOR_TYPE_RGB888_PACKED:
            src[(dy + y1) * src_w * channels + (dx + x1) * channels + 0] =
                color.r;
            src[(dy + y1) * src_w * channels + (dx + x1) * channels + 1] =
                color.g;
            src[(dy + y1) * src_w * channels + (dx + x1) * channels + 2] =
                color.b;
            break;
        case nv::COLOR_TYPE_BGR888_PLANAR:
            src[0 * src_h * src_w + (dy + y1) * src_w + (dx + x1)] = color.b;
            src[1 * src_h * src_w + (dy + y1) * src_w + (dx + x1)] = color.g;
            src[2 * src_h * src_w + (dy + y1) * src_w + (dx + x1)] = color.r;
            break;
        case nv::COLOR_TYPE_RGB888_PLANAR:
            src[0 * src_h * src_w + (dy + y1) * src_w + (dx + x1)] = color.r;
            src[1 * src_h * src_w + (dy + y1) * src_w + (dx + x1)] = color.g;
            src[2 * src_h * src_w + (dy + y1) * src_w + (dx + x1)] = color.b;
            break;
        default:
            break;
        }
    }
}

__global__ void rectangle_kernel(const int nbThreads, unsigned char *src,
                                 int channels, int src_h, int src_w,
                                 nv::BoundingBox bbox,
                                 nv::ColorType src_color_type,
                                 nv::Color color) {
    CUDA_KERNEL_LOOP(idx, nbThreads) {
        int dx = idx % (bbox.x2 - bbox.x1 + 1);
        int dy = idx / (bbox.x2 - bbox.x1 + 1);
        dx += bbox.x1;
        dy += bbox.y1;
        switch (src_color_type) {
        case nv::COLOR_TYPE_BGR888_PACKED:
            src[dy * src_w * channels + dx * channels + 0] =
                CLIP(0.5 * color.b +
                         0.5 * src[dy * src_w * channels + dx * channels + 0],
                     255);
            src[dy * src_w * channels + dx * channels + 1] =
                CLIP(0.5 * color.g +
                         0.5 * src[dy * src_w * channels + dx * channels + 1],
                     255);
            src[dy * src_w * channels + dx * channels + 2] =
                CLIP(0.5 * color.r +
                         0.5 * src[dy * src_w * channels + dx * channels + 2],
                     255);
            break;
        case nv::COLOR_TYPE_RGB888_PACKED:
            src[dy * src_w * channels + dx * channels + 0] =
                CLIP(0.5 * color.r +
                         0.5 * src[dy * src_w * channels + dx * channels + 0],
                     255);
            src[dy * src_w * channels + dx * channels + 1] =
                CLIP(0.5 * color.g +
                         0.5 * src[dy * src_w * channels + dx * channels + 1],
                     255);
            src[dy * src_w * channels + dx * channels + 2] =
                CLIP(0.5 * color.b +
                         0.5 * src[dy * src_w * channels + dx * channels + 2],
                     255);
            break;
        case nv::COLOR_TYPE_BGR888_PLANAR:
            src[0 * src_h * src_w + dy * src_w + dx] = CLIP(
                0.5 * color.b + 0.5 * src[0 * src_h * src_w + dy * src_w + dx],
                255);
            src[1 * src_h * src_w + dy * src_w + dx] = CLIP(
                0.5 * color.g + 0.5 * src[1 * src_h * src_w + dy * src_w + dx],
                255);
            src[2 * src_h * src_w + dy * src_w + dx] = CLIP(
                0.5 * color.r + 0.5 * src[2 * src_h * src_w + dy * src_w + dx],
                255);
            break;
        case nv::COLOR_TYPE_RGB888_PLANAR:
            src[0 * src_h * src_w + dy * src_w + dx] = CLIP(
                0.5 * color.r + 0.5 * src[0 * src_h * src_w + dy * src_w + dx],
                255);
            src[1 * src_h * src_w + dy * src_w + dx] = CLIP(
                0.5 * color.g + 0.5 * src[1 * src_h * src_w + dy * src_w + dx],
                255);
            src[2 * src_h * src_w + dy * src_w + dx] = CLIP(
                0.5 * color.b + 0.5 * src[2 * src_h * src_w + dy * src_w + dx],
                255);
            break;
        default:
            break;
        }
    }
}

void line_hv(nv::Image &src, const nv::Point &p1, const nv::Point &p2,
             const nv::Color &color) {
    // clip
    int x1 = std::max(p1.x, 0);
    int y1 = std::max(p1.y, 0);
    int x2 = std::min(p2.x, src.w() - 1);
    int y2 = std::min(p2.y, src.h() - 1);
    int line_w = x2 - x1 + 1;
    int line_h = y2 - y1 + 1;
    const int nbThreads = line_w * line_h;
    line_hv_kernel<<<CUDA_GET_BLOCKS(nbThreads), CUDA_NUM_THREADS>>>(
        nbThreads, (unsigned char *)(src.gpu_data), src.channels(), src.h(),
        src.w(), src.colorType, line_h, line_w, x1, y1, color);
    CUDACHECK(cudaDeviceSynchronize());
    CUDACHECK(cudaGetLastError());
}

void circle(nv::Image &src, const nv::Point &p1, int radius,
            const nv::Color &color, int thickness) {}

void rectangle(nv::Image &src, const nv::BoundingBox &bbox,
               const nv::Color &color, int thickness) {
    if (thickness > 0) {
        line_hv(src, nv::Point(bbox.x1, bbox.y1),
                nv::Point(bbox.x1 - 1 + thickness, bbox.y2), color);
        line_hv(src, nv::Point(bbox.x1, bbox.y1),
                nv::Point(bbox.x2, bbox.y1 - 1 + thickness), color);
        line_hv(src, nv::Point(bbox.x1, bbox.y2 + 1 - thickness),
                nv::Point(bbox.x2, bbox.y2), color);
        line_hv(src, nv::Point(bbox.x2 + 1 - thickness, bbox.y1),
                nv::Point(bbox.x2, bbox.y2), color);
    } else {
        // thickness = 3;
        // nv::BoundingBox bbox2(bbox.x1 + thickness - 1, bbox.y1, bbox.x2 + 1 -
        // thickness, bbox.y2);
        const int nbThreads = bbox.w() * bbox.h();
        rectangle_kernel<<<CUDA_GET_BLOCKS(nbThreads), CUDA_NUM_THREADS>>>(
            nbThreads, (unsigned char *)(src.gpu_data), src.channels(), src.h(),
            src.w(), bbox, src.colorType, color);
        CUDACHECK(cudaDeviceSynchronize());
        CUDACHECK(cudaGetLastError());
    }
}
}  // namespace nv