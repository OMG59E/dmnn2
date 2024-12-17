/*
 * @Author: xingwg
 * @Date: 2024-12-09 15:27:39
 * @LastEditTime: 2024-12-13 16:45:17
 * @FilePath: /dmnn2/src/codecs/ffmpeg_demuxer.cc
 * @Description:
 *
 * Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */

#include "codecs/ffmpeg_demuxer.h"

namespace nv {
FFmpegDemuxer::~FFmpegDemuxer() {
    if (packet_)
        av_packet_free(&packet_);
    if (packet_filtered_)
        av_packet_free(&packet_filtered_);
    if (bsf_ctx_)
        av_bsf_free(&bsf_ctx_);
    if (input_ctx_)
        avformat_close_input(&input_ctx_);
    if (data_with_headr_)
        av_free(data_with_headr_);
}

int FFmpegDemuxer::open(const std::string &video_path) {
    av_log_set_level(AV_LOG_FATAL);
    AVDictionary *options{nullptr};
    av_dict_set(&options, "rtsp_transport", "tcp",
                0);  // 以udp方式打开，如果以tcp方式打开将udp替换为tcp
    av_dict_set(&options, "stimeout", "2000000",
                0);  // 设置超时断开连接时间，单位微秒
    av_dict_set(&options, "max_delay", "1000000", 0);  // 设置最大时延
    av_dict_set(&options, "correct_ts_overflow", "1", 0);
    if (0 != avformat_open_input(&input_ctx_, video_path.c_str(), nullptr,
                                 &options)) {
        LOG_ERROR("Cannot open input file: {}", video_path);
        return -1;
    }
    av_dict_free(&options);
    if (0 != avformat_find_stream_info(input_ctx_, nullptr)) {
        LOG_ERROR("Cannot find stream info");
        return -1;
    }
    video_stream_idx_ =
        av_find_best_stream(input_ctx_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (video_stream_idx_ < 0) {
        LOG_ERROR("Cannot find video stream");
        return -1;
    }
    auto video_stream = input_ctx_->streams[video_stream_idx_];
    codec_id_ = video_stream->codecpar->codec_id;
    height_ = video_stream->codecpar->height;
    width_ = video_stream->codecpar->width;
    pixel_format_ = AVPixelFormat(video_stream->codecpar->format);
    frame_rate_ = av_q2d(video_stream->avg_frame_rate);
    // time_base_ = 1000.0f / frame_rate_;
    time_base_ = av_q2d(video_stream->time_base);
    duration_ = input_ctx_->duration == AV_NOPTS_VALUE
                    ? 0
                    : double(input_ctx_->duration) / AV_TIME_BASE;
    total_frames_ = video_stream->nb_frames <= 0
                        ? floor(duration_ * frame_rate_ + 0.5f)
                        : video_stream->nb_frames;
    LOG_INFO("URL: {}", video_path);
    LOG_INFO("Media format: {} ({})", input_ctx_->iformat->long_name,
             input_ctx_->iformat->name);
    LOG_INFO("Stream #{}:", video_stream_idx_);
    LOG_INFO("         Codec: {}", avcodec_get_name(codec_id_));
    LOG_INFO("  Pixel format: {}", av_get_pix_fmt_name(pixel_format_));
    LOG_INFO("         Width: {}", width_);
    LOG_INFO("        Height: {}", height_);
    LOG_INFO("    Frame rate: {}", frame_rate_);
    LOG_INFO("      Duration: {:.2f}s", duration_);
    LOG_INFO("  Total Frames: {}", total_frames_);

    switch (pixel_format_) {
    case AV_PIX_FMT_YUV420P10LE:
    case AV_PIX_FMT_GRAY10LE:
        bit_depth_ = 10;
        chroma_height_ = (height_ + 1) >> 1;
        bpp_ = 2;
        break;
    case AV_PIX_FMT_YUV420P12LE:
        bit_depth_ = 12;
        chroma_height_ = (height_ + 1) >> 1;
        bpp_ = 2;
        break;
    case AV_PIX_FMT_YUV444P10LE:
        bit_depth_ = 10;
        chroma_height_ = height_ << 1;
        bpp_ = 2;
        break;
    case AV_PIX_FMT_YUV444P12LE:
        bit_depth_ = 12;
        chroma_height_ = height_ << 1;
        bpp_ = 2;
        break;
    case AV_PIX_FMT_YUV444P:
        bit_depth_ = 8;
        chroma_height_ = height_ << 1;
        bpp_ = 1;
        break;
    case AV_PIX_FMT_YUV420P:
    case AV_PIX_FMT_YUVJ420P:
    case AV_PIX_FMT_YUVJ422P:
    case AV_PIX_FMT_YUVJ444P:
    case AV_PIX_FMT_GRAY8:
        bit_depth_ = 8;
        chroma_height_ = (height_ + 1) >> 1;
        bpp_ = 1;
        break;
    default:
        LOG_WARNING("ChromaFormat not recognized. Assuming 420");
        pixel_format_ = AV_PIX_FMT_YUV420P;
        bit_depth_ = 8;
        chroma_height_ = (height_ + 1) >> 1;
        bpp_ = 1;
    }

    if (!strcmp(input_ctx_->iformat->long_name, "QuickTime / MOV") ||
        !strcmp(input_ctx_->iformat->long_name, "FLV (Flash Video)") ||
        !strcmp(input_ctx_->iformat->long_name, "Matroska / WebM")) {
        is_mp4_h264_ = codec_id_ == AV_CODEC_ID_H264;
        is_mp4_h265_ = codec_id_ == AV_CODEC_ID_HEVC;
        is_mp4_h265_ = codec_id_ == AV_CODEC_ID_MPEG4;
    }

    if (is_mp4_h264_ || is_mp4_h265_) {
        std::string bsf_name =
            is_mp4_h264_ ? "h264_mp4toannexb" : "hevc_mp4toannexb";
        auto bsf = av_bsf_get_by_name(bsf_name.c_str());
        if (!bsf) {
            LOG_ERROR("Cannot find bsf {}", bsf_name);
            return -1;
        }
        if (0 > av_bsf_alloc(bsf, &bsf_ctx_)) {
            LOG_ERROR("Cannot allocate bsf context");
            return -1;
        }
        avcodec_parameters_copy(bsf_ctx_->par_in, video_stream->codecpar);
        if (0 > av_bsf_init(bsf_ctx_)) {
            LOG_ERROR("Cannot initialize bsf context");
            return -1;
        }
    }

    // Allocate the AVPackets and initialize to default values
    packet_ = av_packet_alloc();
    packet_filtered_ = av_packet_alloc();
    if (!packet_ || !packet_filtered_) {
        LOG_ERROR("AVPacket allocation failed");
        return -1;
    }
    return 0;
}

int FFmpegDemuxer::demux(uint8_t **data, int *size, int64_t *pts,
                         bool use_key) {
    *size = 0;
    if (!input_ctx_) {
        LOG_ERROR("Input context is not initialized");
        return -1;
    }
    if (packet_->data)
        av_packet_unref(packet_);

    while (1) {
        int e = av_read_frame(input_ctx_, packet_);
        if (e == AVERROR(EAGAIN) || e == AVERROR_INVALIDDATA) {
            av_packet_unref(packet_);
            continue;
        } else if (e == AVERROR_EOF) {
            av_packet_unref(packet_);
            return 1;
        } else if (e < 0) {  // 某些视频未知错误，暂先跳过
            // LOG_ERROR("Error reading packet: {}, packet size: {}",
            // av_err2string(e), packet_->size);
            av_packet_unref(packet_);
            continue;
        }
        if (video_stream_idx_ != packet_->stream_index ||
            (use_key && !(packet_->flags & AV_PKT_FLAG_KEY)) ||
            packet_->size <= 0) {
            av_packet_unref(packet_);
            continue;
        }
        break;
    }
    if (is_mp4_h264_ || is_mp4_h265_) {
        if (packet_filtered_->data)
            av_packet_unref(packet_filtered_);

        if (0 > av_bsf_send_packet(bsf_ctx_, packet_)) {
            LOG_ERROR("Error sending packet to bsf");
            return -1;
        }

        if (0 > av_bsf_receive_packet(bsf_ctx_, packet_filtered_)) {
            LOG_ERROR("Error receiving packet from bsf");
            return -1;
        }
        *data = packet_filtered_->data;
        *size = packet_filtered_->size;
        if (pts && packet_filtered_->pts != AV_NOPTS_VALUE)
            *pts = int64_t(packet_filtered_->pts * time_base_ *
                           AV_TIME_BASE);  // us
    } else {
        if (is_mp4_mpeg4_ && total_packet_ == 0) {
            int extra_data_size = input_ctx_->streams[video_stream_idx_]
                                      ->codecpar->extradata_size;
            if (extra_data_size > 0) {
                // extradata contains start codes 00 00 01. Subtract its size
                data_with_headr_ = (uint8_t *)av_malloc(
                    extra_data_size + packet_->size - 3 * sizeof(uint8_t));
                if (!data_with_headr_) {
                    LOG_ERROR("Error allocating memory for extradata");
                    return -1;
                }
                memcpy(
                    data_with_headr_,
                    input_ctx_->streams[video_stream_idx_]->codecpar->extradata,
                    extra_data_size);
                memcpy(data_with_headr_ + extra_data_size, packet_->data + 3,
                       packet_->size - 3 * sizeof(uint8_t));
                *data = data_with_headr_;
                *size = extra_data_size + packet_->size - 3 * sizeof(uint8_t);
            }
        } else {
            *data = packet_->data;
            *size = packet_->size;
        }
        if (pts && packet_->pts != AV_NOPTS_VALUE)
            *pts = int64_t(packet_->pts * time_base_ * AV_TIME_BASE);  // us
    }
    total_packet_++;
    return 0;
}
}  // namespace nv