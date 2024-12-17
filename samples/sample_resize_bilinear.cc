/*
 * @Author: xingwg
 * @Date: 2024-11-19 19:40:15
 * @LastEditTime: 2024-12-17 11:16:06
 * @FilePath: /dmnn2/samples/sample_resize_bilinear.cc
 * @Description:
 *
 * Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#include "argparse/argparse.hpp"
#include "codecs/jpeg_dec.h"
#include "codecs/jpeg_enc.h"
#include "imgproc/resize.h"
#include "logging.h"
#include <filesystem>

int main(int argc, char **argv) {
    InitGoogleLogging();
    argparse::ArgumentParser parser(argv[0], std::string("1.0"));
    parser.add_argument("-i", "--input").help("Set image file").required();
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
    std::string image_file = parser.get<std::string>("--input");
    int device_id = parser.get<int>("--device");

    nv::Image image;
    nv::JpegDecoder jpeg_dec;
    if (0 != jpeg_dec.Decode(image_file, image)) {
        LOG_ERROR("decode image failed");
        return -1;
    }

    std::string savePath =
        std::filesystem::current_path() / "resized_image.jpg";

    nv::Image dst;
    dst.create(image.h() * 2, image.w() * 2, true, image.colorType);
    nv::resize(image, dst);
    nv::JpegEncoder jpeg_enc;
    if (0 != jpeg_enc.Encode(savePath, dst)) {
        LOG_ERROR("encode image failed");
        return -1;
    }
    LOG_INFO("save image to {}", savePath);

    dst.free();
    image.free();
    return 0;
}