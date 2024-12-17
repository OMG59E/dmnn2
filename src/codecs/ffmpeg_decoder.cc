/*
 * @Author: xingwg
 * @Date: 2024-10-24 09:00:21
 * @LastEditTime: 2024-12-13 15:58:14
 * @FilePath: /dmnn2/src/codecs/ffmpeg_decoder.cc
 * @Description:
 *
 * Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#include "codecs/ffmpeg_decoder.h"
#include "logging.h"
#include <cstdlib>

std::string av_err2string(int err_code) {
    char err_buf[128];
    if (av_strerror(err_code, err_buf, sizeof(err_buf)) < 0) {
        snprintf(err_buf, sizeof(err_buf), "Unknown error");
    }
    return std::string(err_buf);
}

namespace nv {
FFmpegDecoder::~FFmpegDecoder() {
    av_buffer_unref(&video_codec_ctx_->hw_device_ctx);
    avcodec_free_context(&video_codec_ctx_);
    avformat_close_input(&input_ctx_);
    av_frame_free(&frame_);
    av_packet_free(&packet_);
    if (use_gpu_) {
        CUDA_FREE(buffer_);
    } else {
        CUDA_HOST_FREE(buffer_);
    }
}

int FFmpegDecoder::open(const std::string &video_path, int device_id) {
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

    AVCodec *video_codec = nullptr;
    video_stream_idx_ = av_find_best_stream(input_ctx_, AVMEDIA_TYPE_VIDEO, -1,
                                            -1, &video_codec, 0);
    if (video_stream_idx_ < 0) {
        LOG_ERROR("Cannot find video stream");
        return -1;
    }

    video_codec_ctx_ = avcodec_alloc_context3(video_codec);
    if (!video_codec_ctx_) {
        LOG_ERROR("Cannot allocate video codec context");
        return -1;
    }
    video_codec_ctx_->thread_count = 0;

    // update
    if (avcodec_parameters_to_context(
            video_codec_ctx_,
            input_ctx_->streams[video_stream_idx_]->codecpar) < 0) {
        LOG_ERROR("Cannot copy video codec parameters to context");
        return -1;
    }

    // 不加，可能某些视频无法获取pts
    video_codec_ctx_->framerate.num =
        input_ctx_->streams[video_stream_idx_]->avg_frame_rate.num;
    video_codec_ctx_->framerate.den =
        input_ctx_->streams[video_stream_idx_]->avg_frame_rate.den;

    if (device_id >= 0) {
        // 检查ffmpeg支持的硬件类型
        AVHWDeviceType type = AV_HWDEVICE_TYPE_NONE;
        if (type == AV_HWDEVICE_TYPE_NONE) {
            while ((type = av_hwdevice_iterate_types(type)) !=
                   AV_HWDEVICE_TYPE_NONE)
                LOG_INFO("Available device type: {}",
                         av_hwdevice_get_type_name(type));
        }
        // Set up hardware decoder if possible
        video_codec_ctx_->get_format = [](AVCodecContext *ctx,
                                          const enum AVPixelFormat *pix_fmts) {
            for (const enum AVPixelFormat *p = pix_fmts; *p != -1; p++) {
                if (*p == AV_PIX_FMT_CUDA)
                    return *p;
            }
            LOG_ERROR("Failed to get HW surface format.");
            return AV_PIX_FMT_NONE;
        };
        if (av_hwdevice_ctx_create(
                &video_codec_ctx_->hw_device_ctx, AV_HWDEVICE_TYPE_CUDA,
                std::to_string(device_id).c_str(), nullptr, 0) < 0) {
            LOG_ERROR("Hardware acceleration not available, falling back to "
                      "software decoding.");
            use_gpu_ = false;
            video_codec_ctx_->get_format = nullptr;  // 不使用自定义格式选择回调
        }
    } else {
        use_gpu_ = false;
    }

    if (use_gpu_)
        CUDACHECK(cudaSetDevice(device_id));

    if (avcodec_open2(video_codec_ctx_, video_codec, nullptr) < 0) {
        LOG_ERROR("Cannot open video codec");
        return -1;
    }
    auto video_stream = input_ctx_->streams[video_stream_idx_];
    codec_id_ = video_stream->codecpar->codec_id;
    height_ = video_stream->codecpar->height;
    width_ = video_stream->codecpar->width;
    pixel_format_ = AVPixelFormat(video_stream->codecpar->format);
    frame_rate_ = av_q2d(video_stream->avg_frame_rate);
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

    packet_ = av_packet_alloc();
    if (!packet_) {
        LOG_ERROR("Cannot allocate packet");
        return -1;
    }

    frame_ = av_frame_alloc();
    if (!frame_) {
        LOG_ERROR("Cannot allocate frame");
        return -1;
    }
    return 0;
}

std::string FFmpegDecoder::decoder_name() const {
    switch (input_ctx_->streams[video_stream_idx_]->codecpar->codec_id) {
    case AV_CODEC_ID_H264:
        return "h264_cuvid";
    case AV_CODEC_ID_HEVC:
        return "hevc_cuvid";
    case AV_CODEC_ID_MPEG4:
        return "mpeg4_cuvid";
    case AV_CODEC_ID_MPEG2VIDEO:
        return "mpeg2_cuvid";
    case AV_CODEC_ID_MPEG1VIDEO:
        return "mpeg1_cuvid";
    case AV_CODEC_ID_MJPEG:
        return "mjpeg_cuvid";
    case AV_CODEC_ID_VC1:
        return "vc1_cuvid";
    case AV_CODEC_ID_AV1:
        return "av1_cuvid";
    case AV_CODEC_ID_VP8:
        return "vp8_cuvid";
    case AV_CODEC_ID_VP9:
        return "vp9_cuvid";
    default:
        LOG_WARNING("Not found hardware codec {}",
                    avcodec_get_name(video_codec_ctx_->codec_id));
        return "";
    }
}

nv::ColorType FFmpegDecoder::pixel_format() const {
    switch (pixel_format_) {
    case AV_PIX_FMT_YUV420P:
    case AV_PIX_FMT_YUVJ420P:
        return nv::COLOR_TYPE_YUV420P_YU12;
    case AV_PIX_FMT_NV12:
        return nv::COLOR_TYPE_YUV420SP_NV12;
    case AV_PIX_FMT_NV21:
        return nv::COLOR_TYPE_YUV420SP_NV21;
    case AV_PIX_FMT_YUV422P:
    case AV_PIX_FMT_YUVJ422P:
        return nv::COLOR_TYPE_YUV422P;
    case AV_PIX_FMT_YUV444P:
    case AV_PIX_FMT_YUVJ444P:
        return nv::COLOR_TYPE_YUV444P;
    case AV_PIX_FMT_BGRA:
    case AV_PIX_FMT_BGR0:
        return nv::COLOR_TYPE_BGRA8888_PACKED;
    case AV_PIX_FMT_RGBA:
    case AV_PIX_FMT_RGB0:
        return nv::COLOR_TYPE_RGBA8888_PACKED;
    case AV_PIX_FMT_RGB24:
        return nv::COLOR_TYPE_RGB888_PACKED;
    case AV_PIX_FMT_BGR24:
        return nv::COLOR_TYPE_BGR888_PACKED;
    case AV_PIX_FMT_GRAY8:
        return nv::COLOR_TYPE_GRAY;
    default:
        LOG_FATAL("Unsupported pixel format: {}",
                  av_get_pix_fmt_name(pixel_format_));
        break;
    }
}

int FFmpegDecoder::read(nv::Frame &frame, bool use_key) {
    while (buffer_frame_num_ == 0) {
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
        e = avcodec_send_packet(video_codec_ctx_, packet_);
        if (e == AVERROR(EAGAIN)) {
            LOG_WARNING("Decoder is busy, try avcodec_receive_frame first");
        } else if (e == AVERROR_INVALIDDATA) {
            LOG_WARNING("Invalid data in packet, size: {}", packet_->size);
            av_packet_unref(packet_);
            continue;
        } else if (e == AVERROR_EOF) {
            LOG_WARNING(
                "Decoder has been flushed, no more packets can be sent");
        } else if (e == AVERROR(EINVAL)) {
            LOG_ERROR("Invalid operation: decoder not opened or wrong usage");
            av_packet_unref(packet_);
            return e;
        } else if (e == AVERROR(ENOMEM)) {
            LOG_ERROR("Not enough memory");
            av_packet_unref(packet_);
            return e;
        } else if (e < 0) {
            LOG_ERROR("Error sending frame: {}", av_err2string(e));
            av_packet_unref(packet_);
            return e;
        }

        while (1) {
            e = avcodec_receive_frame(video_codec_ctx_, frame_);
            if (e >= 0) {  // success
                if (buffer_frame_num_ >= MAX_BUFFER_FRAME) {
                    av_frame_unref(frame_);
                    LOG_ERROR("Buffer full, will drop frame");
                    break;
                }
                if (!buffer_) {
                    if (use_gpu_) {
                        AVHWFramesContext *frames_ctx =
                            (AVHWFramesContext *)(frame_->hw_frames_ctx->data);
                        pixel_format_ =
                            frames_ctx
                                ->sw_format;  // 硬解码无法提前获取像素格式，更新硬解码实际像素格式后分配buffer)
                        CUDACHECK(cudaMalloc(&buffer_, MAX_BUFFER_FRAME *
                                                           frame_size_bytes()));
                        LOG_INFO(
                            "atual pixel format: {}, {}",
                            av_get_pix_fmt_name(pixel_format_),
                            av_get_pix_fmt_name(AVPixelFormat(frame_->format)));
                    } else {
                        pixel_format_ = AVPixelFormat(frame_->format);
                        LOG_INFO("atual pixel format: {}",
                                 av_get_pix_fmt_name(pixel_format_));
                        CUDACHECK(cudaMallocHost(
                            &buffer_, MAX_BUFFER_FRAME * frame_size_bytes()));
                    }
                }
                switch (pixel_format()) {
                case nv::COLOR_TYPE_YUV420P_YU12:
                case nv::COLOR_TYPE_YUV420P_YV12:
                    CUDACHECK(cudaMemcpy(buffer_ + buffer_frame_num_ *
                                                       frame_size_bytes(),
                                         frame_->data[0], height() * width(),
                                         use_gpu_ ? cudaMemcpyDeviceToDevice
                                                  : cudaMemcpyHostToHost));
                    CUDACHECK(cudaMemcpy(
                        buffer_ + buffer_frame_num_ * frame_size_bytes() +
                            height() * width(),
                        frame_->data[1], height() * width() / 4,
                        use_gpu_ ? cudaMemcpyDeviceToDevice
                                 : cudaMemcpyHostToHost));
                    CUDACHECK(cudaMemcpy(
                        buffer_ + buffer_frame_num_ * frame_size_bytes() +
                            height() * width() * 5 / 4,
                        frame_->data[2], height() * width() / 4,
                        use_gpu_ ? cudaMemcpyDeviceToDevice
                                 : cudaMemcpyHostToHost));
                    break;
                case nv::COLOR_TYPE_YUV420SP_NV12:
                case nv::COLOR_TYPE_YUV420SP_NV21:
                    CUDACHECK(cudaMemcpy(buffer_ + buffer_frame_num_ *
                                                       frame_size_bytes(),
                                         frame_->data[0], height() * width(),
                                         use_gpu_ ? cudaMemcpyHostToDevice
                                                  : cudaMemcpyHostToHost));
                    CUDACHECK(cudaMemcpy(
                        buffer_ + buffer_frame_num_ * frame_size_bytes() +
                            height() * width(),
                        frame_->data[1], height() * width() / 2,
                        use_gpu_ ? cudaMemcpyHostToDevice
                                 : cudaMemcpyHostToHost));
                    break;
                default:
                    LOG_FATAL("Unsupported pixel format: {}",
                              av_get_pix_fmt_name(pixel_format_));
                    break;
                }
                buffer_frame_num_++;
                continue;
            } else if (e == AVERROR(EAGAIN)) {  // 没有足够码流数据输入
                // LOG_WARNING("Output is not available in this state - user
                // must try to send new input");
                av_frame_unref(frame_);
                break;
            } else if (e == AVERROR_EOF) {  // 缓存帧已被取完
                LOG_WARNING("The decoder has been fully flushed, and there "
                            "will be no more output frames");
                av_packet_unref(packet_);
                av_frame_unref(frame_);
                return 1;  // 结束
            } else if (e < 0) {
                LOG_ERROR("Error receiving frame: {}", av_err2string(e));
                av_packet_unref(packet_);
                av_frame_unref(frame_);
                return e;
            }
        }
        av_packet_unref(packet_);
        continue;
    }
    if (buffer_frame_num_ > 0) {
        // copy from buffer
        frame.own = false;
        frame.colorType = pixel_format();
        frame.dataType = DATA_TYPE_UINT8;
        frame.data = buffer_ + buffer_frame_idx_ * frame_size_bytes();
        frame.height = height();
        frame.width = width();
        frame.idx = total_decoded_frames_;
        frame.timestamp = total_decoded_frames_ * 1000 / frame_rate();  // ms
        buffer_frame_idx_++;
        total_decoded_frames_++;
        if (buffer_frame_idx_ == buffer_frame_num_) {
            // reset
            buffer_frame_idx_ = 0;
            buffer_frame_num_ = 0;
        }
        return 0;
    } else {
        LOG_ERROR("No frame in buffer, please check your input file");
        return -1;
    }
}

size_t FFmpegDecoder::frame_size_bytes() const {
    auto colorType = pixel_format();
    if (colorType == COLOR_TYPE_GRAY) {
        return video_codec_ctx_->width * video_codec_ctx_->height;
    } else if (colorType == COLOR_TYPE_RGB888_PACKED ||
               colorType == COLOR_TYPE_BGR888_PACKED ||
               colorType == COLOR_TYPE_BGR888_PLANAR ||
               colorType == COLOR_TYPE_RGB888_PLANAR ||
               colorType == COLOR_TYPE_YUV444P) {
        return video_codec_ctx_->width * video_codec_ctx_->height * 3;
    } else if (colorType == COLOR_TYPE_YUV420SP_NV12 ||
               colorType == COLOR_TYPE_YUV420SP_NV21 ||
               colorType == COLOR_TYPE_YUV420P_YV12 ||
               colorType == COLOR_TYPE_YUV420P_YU12) {
        return video_codec_ctx_->width * video_codec_ctx_->height * 3 / 2;
    } else if (colorType == COLOR_TYPE_YUV422P) {
        return video_codec_ctx_->width * video_codec_ctx_->height * 2;
    } else if (colorType == COLOR_TYPE_RGBA8888_PACKED ||
               colorType == COLOR_TYPE_BGRA8888_PACKED) {
        return video_codec_ctx_->width * video_codec_ctx_->height * 4;
    } else {
        LOG_FATAL("unknown color type");
    }
}
}  // namespace nv