/*
 * @Author: xingwg
 * @Date: 2024-10-09 13:58:23
 * @LastEditTime: 2024-12-13 15:17:14
 * @FilePath: /dmnn2/tools/onnx2trt.cc
 * @Description:
 *
 * Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#include "argparse/argparse.hpp"
#include "calibrator.h"
#include "common.hpp"
#include "error_check.h"
#include "onnx-ml.pb.h"
#include "trt_logger.h"
#include <NvInferPlugin.h>
#include <NvOnnxParser.h>
#include <cassert>
#include <fstream>
#include <iostream>
#include <string.h>
#include <string>
#include <vector>

static TrtLogger gLogger(ILogger::Severity::kVERBOSE);

struct Parameters {
    std::string model;
    std::string engine;
    std::string calibrationData;
    std::string calibrationCache{"CalibrationTable"};
    DataType dataType{DataType::kHALF};
    int device{0};
    int batchSize{1};
    int workspaceSize{512};
    int iterations{1};
    int avgRuns{1};
    int calibBatchSize{1};
    int calibMaxBatchIdx{100};
    int DLACore{-1};
} gParams;

bool onnxToTrtModel(IInt8Calibrator *calibrator,
                    nvinfer1::IHostMemory **trtModelStream) {
    onnx::ModelProto _the_onnx_model;
    onnx::ModelProto &onnx_model = _the_onnx_model;
    bool is_binary =
        common::ParseFromFile_WAR(&onnx_model, gParams.model.c_str());
    if (!is_binary &&
        !common::ParseFromTextFile(&onnx_model, gParams.model.c_str())) {
        LOG_ERROR("Failed to parse ONNX model");
        return false;
    }
    if (onnx_model.ir_version() > onnx::IR_VERSION) {
        LOG_WARNING("WARNING: ONNX model has a newer ir_version ({}) than this "
                    "parser was built against ({})",
                    common::onnx_ir_version_string(onnx_model.ir_version()),
                    common::onnx_ir_version_string(onnx::IR_VERSION));
    }
    const auto networkFlags =
        1U << static_cast<uint32_t>(
            nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto builder = nvinfer1::createInferBuilder(gLogger);
    auto network = builder->createNetworkV2(networkFlags);
    auto parser = nvonnxparser::createParser(*network, gLogger);
    int32_t verbosity = 2;  // ILogger::Severity::kWARNING
    if (!parser->parseFromFile(gParams.model.c_str(), verbosity)) {
        LOG_ERROR("Failed to parse onnx file");
        return false;
    }
    if (gParams.dataType == DataType::kINT8 &&
        !builder->platformHasFastInt8()) {
        LOG_ERROR("This GPU does not support int8");
        return false;
    }
    if (gParams.dataType == DataType::kINT8) {
        calibrator = new nv::Int8EntropyCalibrator(
            gParams.calibrationData, network, gParams.calibBatchSize,
            gParams.calibMaxBatchIdx);
    }
    if (gParams.dataType == DataType::kHALF &&
        !builder->platformHasFastFp16()) {
        LOG_ERROR("This GPU does not support FP16");
        return false;
    }
    IBuilderConfig *config = builder->createBuilderConfig();
    if (gParams.dataType == DataType::kINT8) {
        config->setAvgTimingIterations(gParams.avgRuns);
        config->setMinTimingIterations(gParams.iterations);
        config->setFlag(BuilderFlag::kDEBUG);
        config->setFlag(BuilderFlag::kINT8);
        config->setFlag(BuilderFlag::kFP16);
        config->setInt8Calibrator(calibrator);
    }
    if (gParams.dataType == DataType::kHALF)
        config->setFlag(BuilderFlag::kFP16);
    if (gParams.DLACore >= 0) {
        LOG_ASSERT(gParams.DLACore < builder->getNbDLACores());
        config->setDefaultDeviceType(DeviceType::kDLA);
        config->setDLACore(gParams.DLACore);
        config->setFlag(BuilderFlag::kSTRICT_TYPES);
        config->setFlag(BuilderFlag::kGPU_FALLBACK);
        if (gParams.batchSize > builder->getMaxDLABatchSize()) {
            LOG_WARNING("Requested batch size {} is greater than the max DLA "
                        "batch size of {}. Reducing batch size accordingly.",
                        gParams.batchSize, builder->getMaxDLABatchSize());
            gParams.batchSize = builder->getMaxDLABatchSize();
        }
    }
    IOptimizationProfile *profile = builder->createOptimizationProfile();
    for (uint8_t i = 0; i < network->getNbInputs(); i++) {
        auto *input = network->getInput(i);
        auto name = input->getName();
        auto const dims = input->getDimensions();
        auto const isDynamicInput =
            std::any_of(dims.d, dims.d + dims.nbDims,
                        [](int32_t dim) { return dim == -1; });
        if (isDynamicInput) {
            LOG_FATAL("Not support dynamic input shape yet");
        } else {
            profile->setDimensions(name, OptProfileSelector::kMIN, dims);
            profile->setDimensions(name, OptProfileSelector::kOPT, dims);
            profile->setDimensions(name, OptProfileSelector::kMAX, dims);
        }
    }
    config->setMaxWorkspaceSize(static_cast<size_t>(gParams.workspaceSize)
                                << 20);
    if (-1 == config->addOptimizationProfile(profile)) {
        LOG_ERROR("Failed to addOptimizationProfile");
        return false;
    }
    ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
    LOG_ASSERT(engine);
    // serialize the engine, then close everything down
    (*trtModelStream) = engine->serialize();
    // we don't need the network anymore, and we can destroy the parser
    network->destroy();
    parser->destroy();
    engine->destroy();
    config->destroy();
    builder->destroy();
    return true;
}

bool parseArgs(int argc, char **argv) {
    argparse::ArgumentParser parser(argv[0], std::string("1.0"));
    parser.add_argument("--model").help("Onnx model file").required();
    parser.add_argument("--engine")
        .help("Engine file to serialize to")
        .required();
    parser.add_argument("--batch")
        .help("Set batch size. For dynamic onnx")
        .default_value(1)
        .scan<'i', int>();
    parser.add_argument("--dtype")
        .help("Run in precision mode. Support fp32/fp16/int8")
        .default_value(std::string("fp16"));
    parser.add_argument("--device")
        .help("Set cuda device to N")
        .default_value(0)
        .scan<'i', int>();
    parser.add_argument("--DLACore")
        .help("Set DLACore to N")
        .default_value(-1)
        .scan<'i', int>();
    parser.add_argument("--iter")
        .help("Run N iterations")
        .default_value(1)
        .scan<'i', int>();
    parser.add_argument("--avg_iter")
        .help("Set the number of averaging iterations used when timing layers")
        .default_value(1)
        .scan<'i', int>();
    parser.add_argument("--workspace")
        .help("Set workspace size in MBytes")
        .default_value(512)
        .scan<'i', int>();
    parser.add_argument("--calib_data")
        .help("Set calibration data dir, for int8 mode")
        .default_value("");
    parser.add_argument("--calib_batch")
        .help("Set calibration batch")
        .scan<'i', int>()
        .default_value(1);
    parser.add_argument("--calib_max_batch_idx")
        .help("Set calibration max batch idx")
        .scan<'i', int>()
        .default_value(100);

    try {
        parser.parse_args(argc, argv);
    } catch (const std::exception &err) {
        std::cerr << parser << std::endl;
        LOG_ERROR(err.what());
        return false;
    }

    gParams.model = parser.get<std::string>("--model");
    gParams.engine = parser.get<std::string>("--engine");
    gParams.batchSize = parser.get<int>("--batch");
    gParams.iterations = parser.get<int>("--iter");
    gParams.avgRuns = parser.get<int>("--avg_iter");
    gParams.device = parser.get<int>("--device");
    gParams.DLACore = parser.get<int>("--DLACore");
    gParams.workspaceSize = parser.get<int>("--workspace");
    gParams.calibrationData = parser.get<std::string>("--calib_data");
    gParams.calibBatchSize = parser.get<int>("--calib_batch");
    gParams.calibMaxBatchIdx = parser.get<int>("--calib_max_batch_idx");
    auto dataType = parser.get<std::string>("--dtype");
    if (dataType == "int8") {
        gParams.dataType = DataType::kINT8;
    } else if (dataType == "fp16") {
        gParams.dataType = DataType::kHALF;
    } else {
        gParams.dataType = DataType::kFLOAT;
    }
    return true;
}

int main(int argc, char **argv) {
    InitGoogleLogging("./logs/onnx2trt.log", 10 * 1024 * 1024, 1, true);
    if (!parseArgs(argc, argv))
        return -1;
    CUDACHECK(cudaSetDevice(gParams.device));
    initLibNvInferPlugins(&gLogger, "");
    nv::Int8EntropyCalibrator *calibrator{nullptr};
    IHostMemory *trtModelStream{nullptr};
    if (!onnxToTrtModel(calibrator, &trtModelStream)) {
        LOG_ERROR("Failed to convrt onnx to trt");
        return -1;
    }
    std::ofstream trtModelFile(gParams.engine.c_str(), std::ios::binary);
    trtModelFile.write((char *)trtModelStream->data(), trtModelStream->size());
    LOG_INFO("Convert model to tensor model cache: {} completed.",
             gParams.engine.c_str());
    trtModelFile.close();
    trtModelStream->destroy();
    SAFE_FREE(calibrator);
    return 0;
}