/***
 * @Author: xingwg
 * @Date: 2024-10-22 14:56:37
 * @LastEditTime: 2024-10-22 15:32:47
 * @FilePath: /dmnn2/include/imgproc/draw.h
 * @Description:
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#pragma once
#include "base_types.h"
#include "logging.h"

namespace nv {
void DECLSPEC_API circle(nv::Image &src, const nv::Point &p1, int radius,
                         const nv::Color &color, int thickness = 1);
void DECLSPEC_API rectangle(nv::Image &src, const nv::BoundingBox &bbox,
                            const nv::Color &color, int thickness = 1);
}  // namespace nv