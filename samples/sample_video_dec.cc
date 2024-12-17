/*
 * @Author: xingwg
 * @Date: 2024-10-25 11:09:40
 * @LastEditTime: 2024-12-17 11:12:52
 * @FilePath: /dmnn2/samples/sample_video_dec.cc
 * @Description:
 *
 * Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#include "argparse/argparse.hpp"
#include "codecs/video_dec.h"
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

    CUcontext ctx;
    cuInit(0);
    cuCtxCreate(&ctx, 0, device_id);
    nv::VideoDecoder dec;
    if (0 != dec.open(video_path, device_id, ctx)) {
        LOG_ERROR("Failed to open video file {}", video_path.c_str());
        return -1;
    }

    nv::Frame frame;
    while (true) {
        int e = dec.read(frame);
        if (e < 0) {
            LOG_ERROR("Failed to read frame");
            break;
        }
        if (e == 1) {
            LOG_INFO("End of file");
            break;
        }
    }
    return 0;
}