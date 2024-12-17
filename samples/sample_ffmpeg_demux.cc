/*
 * @Author: xingwg
 * @Date: 2024-10-25 11:09:40
 * @LastEditTime: 2024-12-17 10:26:59
 * @FilePath: /dmnn2/samples/sample_ffmpeg_demux.cc
 * @Description:
 *
 * Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#include "argparse/argparse.hpp"
#include "codecs/ffmpeg_demuxer.h"
#include "logging.h"

int main(int argc, char **argv) {
    InitGoogleLogging();
    argparse::ArgumentParser parser(argv[0], std::string("1.0"));
    parser.add_argument("-i", "--input").help("Set video file").required();
    try {
        parser.parse_args(argc, argv);
    } catch (const std::exception &err) {
        std::cerr << parser << std::endl;
        LOG_ERROR(err.what());
        return -1;
    }
    std::string video_path = parser.get<std::string>("--input");

    nv::FFmpegDemuxer demux;
    if (0 != demux.open(video_path)) {
        LOG_ERROR("Failed to open video file {}", video_path.c_str());
        return -1;
    }

    uint8_t *data = nullptr;
    int size = 0;
    while (true) {
        int e = demux.demux(&data, &size);
        if (e < 0) {
            LOG_ERROR("Failed to read packet");
            break;
        }
        if (e == 1) {
            LOG_INFO("End of file");
            break;
        }
    }
    return 0;
}