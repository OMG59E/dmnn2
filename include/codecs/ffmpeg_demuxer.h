/***
 * @Author: xingwg
 * @Date: 2024-11-01 10:56:33
 * @LastEditTime: 2024-11-13 15:47:19
 * @FilePath: /dmnn2/include/codecs/ffmpeg_demuxer.h
 * @Description:
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#pragma once
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avassert.h>
#include <libavutil/hwcontext.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
}
#include "cuviddec.h"
#include "logging.h"
#include <string>

namespace nv {
class FFmpegDemuxer {
public:
    FFmpegDemuxer() = default;
    ~FFmpegDemuxer();
    int open(const std::string &video_path);
    int demux(uint8_t **data, int *size, int64_t *pts = nullptr,
              bool use_key = false);
    AVCodecID video_codec() const { return codec_id_; }
    AVPixelFormat pixel_format() const { return pixel_format_; }
    AVCodecParameters *codec_parameters() const {
        return input_ctx_->streams[video_stream_idx_]->codecpar;
    }
    int bit_depth() const { return bit_depth_; }
    int bpp() const { return bpp_; }
    int height() const { return height_; }
    int width() const { return width_; }
    float frame_rate() const { return frame_rate_; }
    int64_t total_frames() const { return total_frames_; }

private:
    AVPixelFormat pixel_format_{AV_PIX_FMT_NONE};
    AVCodecID codec_id_{AV_CODEC_ID_NONE};
    AVFormatContext *input_ctx_{nullptr};
    AVBSFContext *bsf_ctx_{nullptr};
    AVPacket *packet_{nullptr};
    AVPacket *packet_filtered_{nullptr};
    double time_base_{0};
    double frame_rate_{0};
    double duration_{0};
    int video_stream_idx_{0};
    int height_{0};
    int width_{0};
    int chroma_height_{0};
    int bit_depth_{0};  //
    int bpp_{0};        // Bytes Per Pixel
    bool is_mp4_h264_{false};
    bool is_mp4_h265_{false};
    bool is_mp4_mpeg4_{false};
    uint8_t *data_with_headr_{nullptr};
    uint64_t total_packet_{0};  // total read packet already
    int64_t total_frames_{0};
};

inline cudaVideoCodec FFmpeg2NvCodecId(AVCodecID codec_id) {
    switch (codec_id) {
    case AV_CODEC_ID_MPEG1VIDEO:
        return cudaVideoCodec_MPEG1;
    case AV_CODEC_ID_MPEG2VIDEO:
        return cudaVideoCodec_MPEG2;
    case AV_CODEC_ID_MPEG4:
        return cudaVideoCodec_MPEG4;
    case AV_CODEC_ID_WMV3:
    case AV_CODEC_ID_VC1:
        return cudaVideoCodec_VC1;
    case AV_CODEC_ID_H264:
        return cudaVideoCodec_H264;
    case AV_CODEC_ID_HEVC:
        return cudaVideoCodec_HEVC;
    case AV_CODEC_ID_VP8:
        return cudaVideoCodec_VP8;
    case AV_CODEC_ID_VP9:
        return cudaVideoCodec_VP9;
    case AV_CODEC_ID_MJPEG:
        return cudaVideoCodec_JPEG;
    case AV_CODEC_ID_AV1:
        return cudaVideoCodec_AV1;
    default:
        return cudaVideoCodec_NumCodecs;
    }
}

inline cudaVideoChromaFormat FFmpeg2NvChromaFormat(AVPixelFormat pix_fmt) {
    switch (pix_fmt) {
    case AV_PIX_FMT_YUV420P:
    case AV_PIX_FMT_YUVJ420P:
    case AV_PIX_FMT_NV12:
    case AV_PIX_FMT_NV21:
    case AV_PIX_FMT_YUVA420P:
    case AV_PIX_FMT_YUV420P16LE:
    case AV_PIX_FMT_YUV420P16BE:
    case AV_PIX_FMT_YUV420P9BE:
    case AV_PIX_FMT_YUV420P9LE:
    case AV_PIX_FMT_YUV420P10BE:
    case AV_PIX_FMT_YUV420P10LE:
        return cudaVideoChromaFormat_420; /**< YUV 4:2:0  */
    case AV_PIX_FMT_YUYV422:
    case AV_PIX_FMT_YUVJ422P:
    case AV_PIX_FMT_YUV422P:
        return cudaVideoChromaFormat_422; /**< YUV 4:2:2  */
    case AV_PIX_FMT_YUV444P:
    case AV_PIX_FMT_YUVJ444P:
        return cudaVideoChromaFormat_444; /**< YUV 4:4:4  */
    default:
        return cudaVideoChromaFormat_Monochrome;
    }
}
}  // namespace nv