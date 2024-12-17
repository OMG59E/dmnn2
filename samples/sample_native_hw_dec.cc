/*
 * @Author: xingwg
 * @Date: 2024-10-25 11:09:40
 * @LastEditTime: 2024-12-17 10:26:34
 * @FilePath: /dmnn2/samples/sample_native_hw_dec.cc
 * @Description:
 *
 * Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#include "argparse/argparse.hpp"
#include "base_types.h"
#include "codecs/ffmpeg_demuxer.h"
#include "codecs/nv_decoder.h"
#include "logging.h"

int main(int argc, char **argv) {
    InitGoogleLogging();
    argparse::ArgumentParser parser(argv[0], std::string("1.0"));
    parser.add_argument("-i", "--input").help("Set video file").required();
    parser.add_argument("-d", "--device")
        .help("Run in which device")
        .default_value(0)
        .scan<'i', int>();
    try {
        parser.parse_args(argc, argv);
    } catch (const std::exception &err) {
        std::cerr << parser << std::endl;
        LOG_ERROR(err.what());
        return -1;
    }
    std::string video_path = parser.get<std::string>("--input");
    int device_id = parser.get<int>("--device");

    nv::FFmpegDemuxer demux;
    if (0 != demux.open(video_path)) {
        LOG_ERROR("Failed to open video file {}", video_path.c_str());
        return -1;
    }

    cuInit(0);
    CUcontext cuContext = nullptr;
    CUdevice cuDevice = 0;
    CUDA_DRVAPI_CALL(cuDeviceGet(&cuDevice, device_id));
    char szDeviceName[80];
    CUDA_DRVAPI_CALL(
        cuDeviceGetName(szDeviceName, sizeof(szDeviceName), cuDevice));
    LOG_INFO("GPU in use: {}", szDeviceName);
    CUDA_DRVAPI_CALL(cuCtxCreate(&cuContext, 0, cuDevice));
    nv::Rect cropRect;
    nv::Dim resizeDim;
    bool bUseDeviceFrame = true;
    bool bLowLatency = true;
    bool bDeviceFramePitched = false;
    bool bExtractUserSEIMessage = false;
    nv::NvDecoder dec(cuContext, bUseDeviceFrame,
                      nv::FFmpeg2NvCodecId(demux.video_codec()), bLowLatency,
                      bDeviceFramePitched, &cropRect, &resizeDim,
                      bExtractUserSEIMessage);

    uint8_t *data = nullptr;
    int size = 0;
    nv::Frame frame;
    bool first = true;
    int64_t pts = 0;
    while (true) {
        int e = demux.demux(&data, &size, &pts);
        if (e < 0) {
            LOG_ERROR("Failed to read packet");
            break;
        }
        if (e == 1) {
            LOG_INFO("End of file");
            break;
        }
        LOG_INFO("pts: {}", pts);
        int nFrameReturned = dec.Decode(data, size);
        if (first && nFrameReturned) {
            first = false;
            LOG_INFO(dec.GetVideoInfo());
        }
    }
    return 0;
}