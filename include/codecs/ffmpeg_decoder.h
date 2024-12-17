/***
 * @Author: xingwg
 * @Date: 2024-10-23 16:58:06
 * @LastEditTime: 2024-11-08 15:54:36
 * @FilePath: /dmnn2/include/codecs/ffmpeg_decoder.h
 * @Description:
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#pragma once
#include <queue>
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avassert.h>
#include <libavutil/hwcontext.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
}
#include "base_types.h"
#include "error_check.h"

#define MAX_BUFFER_FRAME 5

namespace nv {
class DECLSPEC_API FFmpegDecoder {
public:
    FFmpegDecoder() = default;
    ~FFmpegDecoder();

    int open(const std::string &video_path, int device_id);
    int read(nv::Frame &frame, bool use_key = false);
    int height() const { return height_; };
    int width() const { return width_; };
    float frame_rate() const { return frame_rate_; }
    float duration() const { return duration_; }
    int64_t total_frames() const { return total_frames_; }
    int64_t total_decoded_frames() const { return total_decoded_frames_; }

private:
    std::string decoder_name() const;
    size_t frame_size_bytes() const;
    nv::ColorType pixel_format() const;

private:
    bool use_gpu_{false};
    double time_base_{0};
    double frame_rate_{0};
    double duration_{0};
    int video_stream_idx_{0};
    int height_{0};
    int width_{0};
    int64_t total_decoded_frames_{0};
    int64_t total_frames_{0};
    uint32_t buffer_frame_num_{0};
    uint32_t buffer_frame_idx_{0};
    AVPacket *packet_{nullptr};
    AVFrame *frame_{nullptr};  // 存储解码后的帧
    AVPixelFormat pixel_format_{AV_PIX_FMT_NONE};
    AVCodecID codec_id_{AV_CODEC_ID_NONE};
    AVFormatContext *input_ctx_{nullptr};
    AVCodecContext *video_codec_ctx_{nullptr};
    uint8_t *buffer_{nullptr};
};
}  // namespace nv