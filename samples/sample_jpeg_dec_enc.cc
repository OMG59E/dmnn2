/*
 * @Author: xingwg
 * @Date: 2024-10-22 08:37:29
 * @LastEditTime: 2024-12-17 10:26:40
 * @FilePath: /dmnn2/samples/sample_jpeg_dec_enc.cc
 * @Description:
 *
 * Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#include "argparse/argparse.hpp"
#include "base_types.h"
#include "codecs/jpeg_dec.h"
#include "codecs/jpeg_enc.h"
#include "models/yolov6.h"
#include <filesystem>
#include <vector>

int main(int argc, char **argv) {
    InitGoogleLogging();
    argparse::ArgumentParser parser(argv[0], std::string("1.0"));
    parser.add_argument("-i", "--image").help("Set image file").required();
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
    std::string image_file = parser.get<std::string>("--image");
    int device_id = parser.get<int>("--device");

    nv::Image image;
    nv::JpegDecoder jpeg_dec;
    if (0 != jpeg_dec.Decode(image_file, image)) {
        LOG_ERROR("decode image failed");
        return -1;
    }

    std::string savePath =
        std::filesystem::current_path() / "encoded_image.jpg";
    nv::JpegEncoder jpeg_enc;
    if (0 != jpeg_enc.Encode(savePath, image)) {
        LOG_ERROR("encode image failed");
        return -1;
    }
    LOG_INFO("save image to {}");

    image.free();
    return 0;
}