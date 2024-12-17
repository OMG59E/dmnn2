/*
 * @Author: xingwg
 * @Date: 2024-10-21 15:02:54
 * @LastEditTime: 2024-12-17 11:19:10
 * @FilePath: /dmnn2/samples/sample_infer_yolov6.cc
 * @Description:
 *
 * Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#include "argparse/argparse.hpp"
#include "base_types.h"
#include "codecs/jpeg_dec.h"
#include "codecs/jpeg_enc.h"
#include "imgproc/draw.h"
#include "models/yolov6.h"
#include <filesystem>
#include <vector>

int main(int argc, char **argv) {
    InitGoogleLogging();
    argparse::ArgumentParser parser(argv[0], std::string("1.0"));
    parser.add_argument("-m", "--model").help("Trt model file").required();
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
    std::string model_file = parser.get<std::string>("--model");
    std::string image_file = parser.get<std::string>("--image");
    int device_id = parser.get<int>("--device");

    nv::Image image;
    nv::JpegDecoder jpeg_dec;
    if (0 != jpeg_dec.Decode(image_file, image)) {
        LOG_ERROR("decode image failed");
        return -1;
    }

    nv::YoloV6 model;
    if (0 != model.load(model_file, device_id)) {
        LOG_ERROR("load model failed");
        return -1;
    }

    if (0 != model.preprocess({image})) {
        LOG_ERROR("preprocess failed");
        return -1;
    }

    int valid_batch_size = 1;
    if (0 != model.inference(valid_batch_size)) {
        LOG_ERROR("inference failed");
        return -1;
    }

    std::vector<nv::detections_t> detections;
    if (0 != model.postprocess({image}, detections)) {
        LOG_ERROR("postprocess failed");
        return -1;
    }

    for (auto &image_detections : detections) {
        for (auto &detection : image_detections) {
            LOG_INFO("label={:2d} "
                     "score={:.6f} x1={:4d} "
                     "y1={:4d} x2={:4d} y2={:4d}",
                     detection.cls_idx, detection.score, detection.bbox.x1,
                     detection.bbox.y1, detection.bbox.x2, detection.bbox.y2);
            nv::rectangle(image, detection.bbox, nv::Color(255, 0, 0), 3);
        }
    }

    std::string savePath = std::filesystem::current_path() / "res.jpg";
    nv::JpegEncoder jpeg_enc;
    if (0 != jpeg_enc.Encode(savePath, image)) {
        LOG_ERROR("encode image failed");
        return -1;
    }
    LOG_INFO("save image to {}", savePath);

    if (0 != model.unload()) {
        LOG_ERROR("unload failed");
        return -1;
    }

    image.free();
    return 0;
}