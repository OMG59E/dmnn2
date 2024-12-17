/*
 * @Author: xingwg
 * @Date: 2024-11-06 17:30:49
 * @LastEditTime: 2024-11-13 15:49:57
 * @FilePath: /dmnn2/src/codecs/video_dec.cc
 * @Description:
 *
 * Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#include "codecs/video_dec.h"
#include "codecs/ffmpeg_decoder.h"
#include "codecs/ffmpeg_demuxer.h"
#include "codecs/nv_decoder.h"
#include "nvcuvid.h"

namespace nv {
VideoDecoder::~VideoDecoder() {
    if (use_gpu_) {
        auto demuxer = static_cast<FFmpegDemuxer *>(demuxer_);
        auto decoder = static_cast<NvDecoder *>(decoder_);
        SAFE_FREE(demuxer);
        SAFE_FREE(decoder);
    } else {
        auto decoder = static_cast<FFmpegDecoder *>(decoder_);
        SAFE_FREE(decoder);
    }
    demuxer_ = nullptr;
    decoder_ = nullptr;
}

int VideoDecoder::open(const std::string &filename, int device_id,
                       CUcontext ctx) {
    use_gpu_ = false;
    if (device_id >= 0) {
        auto demuxer = new FFmpegDemuxer();
        if (demuxer->open(filename) != 0) {
            LOG_ERROR("Failed to open video/stream: {}", filename);
            return -1;
        }
        demuxer_ = demuxer;
        if (!is_supported()) {
            use_gpu_ = false;
            LOG_WARNING("GPU acceleration not available, falling back to CPU "
                        "decoding.");
            SAFE_FREE(demuxer);
            demuxer_ = nullptr;
        } else {
            use_gpu_ = true;
            demuxer_ = demuxer;
            nv::Rect cropRect;
            nv::Dim resizeDim;
            bool bUseDeviceFrame = true;
            bool bLowLatency = true;
            bool bDeviceFramePitched = false;
            bool bExtractUserSEIMessage = false;
            auto *decoder =
                new NvDecoder(ctx, bUseDeviceFrame,
                              nv::FFmpeg2NvCodecId(demuxer->video_codec()),
                              bLowLatency, bDeviceFramePitched, &cropRect,
                              &resizeDim, bExtractUserSEIMessage);
            decoder_ = decoder;
            total_frames_ = demuxer->total_frames();
        }
    }

    if (!use_gpu_) {
        auto decoder = new FFmpegDecoder();
        if (decoder->open(filename, -1) != 0) {
            LOG_ERROR("Failed to open video/stream: {}", filename);
            return -1;
        }
        decoder_ = decoder;
        total_frames_ = decoder->total_frames();
    }
    return 0;
}

int VideoDecoder::read(nv::Frame &frame, bool use_key) {
    if (use_gpu_) {
        uint8_t *data = nullptr;
        int size = 0;
        int64_t pts = 0;
        auto demuxer = static_cast<FFmpegDemuxer *>(demuxer_);
        auto decoder = static_cast<NvDecoder *>(decoder_);
        while (queue_.empty()) {
            int e = demuxer->demux(&data, &size, &pts, use_key);
            if (e < 0) {
                LOG_ERROR("Failed to read packet");
                return e;
            }
            if (e == 1) {
                LOG_INFO("End of file");
                return 1;
            }
            int frame_returned = decoder->Decode(data, size);
            for (int i = 0; i < frame_returned; ++i) {
                uint8_t *frame_data = decoder->GetFrame(&pts);
                queue_.push(frame_data);
            }
        }
        uint8_t *frame_data = queue_.front();
        frame.own = false;
        frame.data = frame_data;
        frame.height = demuxer->height();
        frame.width = demuxer->width();
        frame.timestamp =
            uint64_t(total_decoded_frames_ * 1e6 / demuxer->frame_rate());
        if (decoder->GetOutputFormat() == cudaVideoSurfaceFormat_NV12) {
            frame.colorType = COLOR_TYPE_YUV420SP_NV12;
            frame.dataType = DATA_TYPE_UINT8;
        } else if (decoder->GetOutputFormat() == cudaVideoSurfaceFormat_P016) {
            frame.colorType = COLOR_TYPE_YUV420SP_NV12;
            frame.dataType = DATA_TYPE_UINT16;
        } else {
            LOG_ERROR("Unsupported output format");
            return -1;
        }
        queue_.pop();
    } else {
        auto decoder = static_cast<FFmpegDecoder *>(decoder_);
        int e = decoder->read(frame);
        if (e < 0) {
            LOG_ERROR("Failed to read frame");
            return e;
        }
        if (e == 1) {
            LOG_INFO("End of file");
            return 1;
        }
        frame.timestamp = uint64_t(total_decoded_frames_ * 1e6 /
                                   decoder->frame_rate());  // us
    }
    frame.idx = total_decoded_frames_;
    total_decoded_frames_++;
    return 0;
}

bool VideoDecoder::is_stream() const {
    // if (input_ctx_->iformat->flags & AVFMT_NOFILE || (fmt_ctx->pb &&
    // fmt_ctx->pb->seekable == 0))
    //     return true;
    return false;
}

int VideoDecoder::height() const {
    if (use_gpu_) {
        auto decoder = static_cast<NvDecoder *>(decoder_);
        return decoder->GetHeight();
    } else {
        auto decoder = static_cast<FFmpegDecoder *>(decoder_);
        return decoder->height();
    }
}

int VideoDecoder::width() const {
    if (use_gpu_) {
        auto decoder = static_cast<NvDecoder *>(decoder_);
        return decoder->GetWidth();
    } else {
        auto decoder = static_cast<FFmpegDecoder *>(decoder_);
        return decoder->width();
    }
}

int64_t VideoDecoder::total_decoded_frames() const {
    return total_decoded_frames_;
}

int64_t VideoDecoder::total_frames() const { return total_frames_; }

bool VideoDecoder::is_supported() const {
    auto demuxer = static_cast<FFmpegDemuxer *>(demuxer_);

    CUVIDDECODECAPS decodecaps;
    memset(&decodecaps, 0, sizeof(decodecaps));

    decodecaps.eCodecType = FFmpeg2NvCodecId(demuxer->video_codec());
    decodecaps.eChromaFormat = FFmpeg2NvChromaFormat(demuxer->pixel_format());
    decodecaps.nBitDepthMinus8 = demuxer->bit_depth() - 8;

    int e = cuvidGetDecoderCaps(&decodecaps);
    if (e != CUDA_SUCCESS) {
        LOG_ERROR("Failed to get decoder caps");
        return false;
    }

    if (!decodecaps.bIsSupported) {
        LOG_ERROR("Codec not supported on this GPU {}",
                  int(CUDA_ERROR_NOT_SUPPORTED));
        return false;
    }

    if ((demuxer->width() > decodecaps.nMaxWidth) ||
        (demuxer->height() > decodecaps.nMaxHeight)) {
        std::ostringstream errorString;
        errorString << std::endl
                    << "Resolution          : " << demuxer->width() << "x"
                    << demuxer->height() << std::endl
                    << "Max Supported (wxh) : " << decodecaps.nMaxWidth << "x"
                    << decodecaps.nMaxHeight << std::endl
                    << "Resolution not supported on this GPU";
        const std::string cErr = errorString.str();
        LOG_ERROR(cErr + "{}", int(CUDA_ERROR_NOT_SUPPORTED));
        return false;
    }

    if ((demuxer->width() >> 4) * (demuxer->height() >> 4) >
        decodecaps.nMaxMBCount) {
        std::ostringstream errorString;
        errorString << std::endl
                    << "MBCount             : "
                    << (demuxer->width() >> 4) * (demuxer->height() >> 4)
                    << std::endl
                    << "Max Supported mbcnt : " << decodecaps.nMaxMBCount
                    << std::endl
                    << "MBCount not supported on this GPU";
        const std::string cErr = errorString.str();
        LOG_ERROR(cErr + "{}", int(CUDA_ERROR_NOT_SUPPORTED));
        return false;
    }

    auto chroma_format = FFmpeg2NvChromaFormat(demuxer->pixel_format());
    cudaVideoSurfaceFormat output_format{cudaVideoSurfaceFormat_NV12};
    // Set the output surface format same as chroma format
    if (chroma_format == cudaVideoChromaFormat_420 ||
        cudaVideoChromaFormat_Monochrome)
        output_format = demuxer->bpp() - 8 ? cudaVideoSurfaceFormat_P016
                                           : cudaVideoSurfaceFormat_NV12;
    else if (chroma_format == cudaVideoChromaFormat_444)
        output_format = demuxer->bpp() - 8 ? cudaVideoSurfaceFormat_YUV444_16Bit
                                           : cudaVideoSurfaceFormat_YUV444;
    else if (chroma_format == cudaVideoChromaFormat_422)
        output_format =
            cudaVideoSurfaceFormat_NV12;  // no 4:2:2 output format supported
                                          // yet so make 420 default

    // Check if output format supported. If not, check falback options
    if (!(decodecaps.nOutputFormatMask & (1 << output_format))) {
        if (decodecaps.nOutputFormatMask & (1 << cudaVideoSurfaceFormat_NV12))
            output_format = cudaVideoSurfaceFormat_NV12;
        else if (decodecaps.nOutputFormatMask &
                 (1 << cudaVideoSurfaceFormat_P016))
            output_format = cudaVideoSurfaceFormat_P016;
        else if (decodecaps.nOutputFormatMask &
                 (1 << cudaVideoSurfaceFormat_YUV444))
            output_format = cudaVideoSurfaceFormat_YUV444;
        else if (decodecaps.nOutputFormatMask &
                 (1 << cudaVideoSurfaceFormat_YUV444_16Bit))
            output_format = cudaVideoSurfaceFormat_YUV444_16Bit;
        else {
            LOG_ERROR("No supported output format found {}",
                      int(CUDA_ERROR_NOT_SUPPORTED));
            return false;
        }
    }
    return true;
}
}  // namespace nv