/*
 * @Author: xingwg
 * @Date: 2024-10-15 14:37:45
 * @LastEditTime: 2024-12-13 17:00:04
 * @FilePath: /dmnn2/tools/nv_run_model.cc
 * @Description:
 *
 * Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#include "argparse/argparse.hpp"
#include "logging.h"
#include "models/net_operator.h"
#include <algorithm>
#include <chrono>

using namespace std::chrono;

float calculateTP99(std::vector<float> &responseTimes) {
    std::sort(responseTimes.begin(), responseTimes.end());
    int N = responseTimes.size();
    int P99_Index = std::ceil(0.99 * N) - 1;
    return responseTimes[P99_Index];
}

float calculateTP999(std::vector<float> &responseTimes) {
    std::sort(responseTimes.begin(), responseTimes.end());
    int N = responseTimes.size();
    int P99_9_Index = std::ceil(0.999 * N) - 1;
    return responseTimes[P99_9_Index];
}

int main(int argc, char **argv) {
    InitGoogleLogging();
    argparse::ArgumentParser parser(argv[0], std::string("1.0"));
    parser.add_argument("-m", "--model").help("Trt model file").required();
    parser.add_argument("-b", "--batch")
        .help("Set batch size")
        .default_value(1)
        .scan<'i', int>();
    parser.add_argument("-w", "--warmup")
        .help("Set warmup")
        .default_value(5)
        .scan<'i', int>();
    parser.add_argument("-r", "--repeat")
        .help("Set repeat inference")
        .default_value(100)
        .scan<'i', int>();
    parser.add_argument("-d", "--device")
        .help("Run in which device")
        .default_value(0)
        .scan<'i', int>();
    parser.add_argument("-l", "--layers")
        .help("Print layer time")
        .default_value(false)
        .implicit_value(true);

    try {
        parser.parse_args(argc, argv);
    } catch (const std::exception &err) {
        std::cerr << parser << std::endl;
        LOG_ERROR(err.what());
        return -1;
    }

    std::string model_file = parser.get<std::string>("--model");
    int batch_size = parser.get<int>("--batch");
    int warmup = parser.get<int>("--warmup");
    int repeat = parser.get<int>("--repeat");
    int device_id = parser.get<int>("--device");
    auto enable_layers = parser["--layers"];

    nv::NetOperator net;
    if (0 != net.load(model_file, device_id)) {
        LOG_ERROR("load model failed");
        return -1;
    }

    float total_latency = 0;
    float min_latency = std::numeric_limits<float>::max();
    float max_latency = std::numeric_limits<float>::min();
    std::vector<float> latencys;
    for (int i = 0; i < warmup + repeat; ++i) {
        high_resolution_clock::time_point t0 = high_resolution_clock::now();
        if (0 != net.inference(batch_size)) {
            LOG_ERROR("inference failed");
            return -1;
        }
        high_resolution_clock::time_point t1 = high_resolution_clock::now();
        if (i < warmup)
            continue;
        duration<float, std::micro> tp = t1 - t0;
        float latency = tp.count() / 1000.0f;
        total_latency += latency;
        latencys.emplace_back(latency);
        if (latency < min_latency)
            min_latency = latency;
        if (latency > max_latency)
            max_latency = latency;
    }
    float ave = total_latency / repeat;
    LOG_INFO("Batch: {}  repeat: {}  warmup: {}", batch_size, repeat, warmup);
    LOG_INFO("Latency min: {:.3f}ms  max: {:.3f}ms  ave: {:.3f}ms  tp99: "
             "{:.3f}ms  tp99.9: {:.3f}ms",
             min_latency, max_latency, ave, calculateTP99(latencys),
             calculateTP999(latencys));

    if (enable_layers == true)
        net.printLayerTimes(repeat);

    if (0 != net.unload()) {
        LOG_ERROR("unload failed");
        return -1;
    }
    return 0;
}