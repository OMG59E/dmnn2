/***
 * @Author: xingwg
 * @Date: 2024-11-06 17:05:58
 * @LastEditTime: 2024-11-12 14:21:27
 * @FilePath: /dmnn2/include/codecs/video_dec.h
 * @Description:
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#pragma once
#include "base_types.h"
#include <queue>

namespace nv {
class VideoDecoder {
public:
    VideoDecoder() = default;
    ~VideoDecoder();
    int open(const std::string &filename, int device_id,
             CUcontext ctx = nullptr);
    int read(nv::Frame &frame, bool use_key = false);
    int height() const;
    int width() const;
    int64_t total_decoded_frames() const;
    int64_t total_frames() const;
    bool is_stream() const;

private:
    bool is_supported() const;

private:
    int64_t total_frames_{0};
    int64_t total_decoded_frames_{0};
    bool use_gpu_{true};
    std::queue<uint8_t *> queue_;
    void *demuxer_{nullptr};
    void *decoder_{nullptr};
};
}  // namespace nv
