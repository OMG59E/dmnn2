/*
 * @Author: xingwg
 * @Date: 2024-10-11 11:13:13
 * @LastEditTime: 2024-10-15 17:08:48
 * @FilePath: /dmnn2/tools/caffe2trt.cc
 * @Description:
 *
 * Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#include "argparse/argparse.hpp"
#include "calibrator.h"
#include "error_check.h"
#include "trt_logger.h"
#include <NvCaffeParser.h>
#include <NvInferPlugin.h>
#include <cassert>
#include <fstream>
#include <iostream>
#include <string.h>
#include <string>
#include <vector>

using namespace nvinfer1;
using namespace nvcaffeparser1;

static TrtLogger gLogger(ILogger::Severity::kVERBOSE);

struct Parameters {
    std::string deployFile;
    std::string modelFile;
    std::string engine;
    std::vector<std::string> output_names;
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

bool parseArgs(int argc, char **argv) {
    argparse::ArgumentParser parser(argv[0], std::string("1.0"));
    parser.add_argument("--deploy").help("Caffe deploy file").required();
    parser.add_argument("--model").help("Caffe model file").required();
    parser.add_argument("--output_names")
        .help("model output names")
        .required()
        .nargs(argparse::nargs_pattern::at_least_one);
    parser.add_argument("--engine")
        .help("Engine file to serialize to")
        .required();
    parser.add_argument("--batch")
        .help("Set batch size")
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
        .default_value(1)
        .scan<'i', int>();
    parser.add_argument("--calib_max_batch_idx")
        .help("Set calibration max batch")
        .default_value(100)
        .scan<'i', int>();

    try {
        parser.parse_args(argc, argv);
    } catch (const std::exception &err) {
        std::cerr << parser << std::endl;
        LOG_ERROR(err.what());
        return false;
    }
    gParams.deployFile = parser.get<std::string>("--deploy");
    gParams.modelFile = parser.get<std::string>("--model");
    gParams.engine = parser.get<std::string>("--engine");
    gParams.output_names =
        parser.get<std::vector<std::string>>("--output_names");
    gParams.batchSize = parser.get<int>("--batch");
    gParams.iterations = parser.get<int>("--iter");
    gParams.avgRuns = parser.get<int>("--avg_iter");
    gParams.device = parser.get<int>("--device");
    gParams.DLACore = parser.get<int>("--DLACore");
    gParams.workspaceSize = parser.get<int>("--workspace");
    gParams.calibrationData = parser.get<std::string>("--calib_data");
    gParams.calibBatchSize = parser.get<int>("--calib_batch");
    gParams.calibMaxBatchIdx = parser.get<int>("--calib_max_batch_idx");
    ;
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

bool caffeToTrtModel(IInt8Calibrator *calibrator,
                     nvinfer1::IHostMemory **trtModelStream) {
    IBuilder *builder = createInferBuilder(gLogger);
    auto networkFlags =
        gParams.batchSize
            ? 0U
            : 1U << static_cast<uint32_t>(
                  nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    // parse the caffe model to populate the network, then set the outputs
    INetworkDefinition *network = builder->createNetworkV2(networkFlags);
    ICaffeParser *parser = createCaffeParser();
    if (gParams.dataType == DataType::kINT8 &&
        !builder->platformHasFastInt8()) {
        LOG_ERROR("The device for INT8 run since its not supported.");
        return false;
    }
    if (gParams.dataType == DataType::kINT8) {
        calibrator = new nv::Int8EntropyCalibrator(
            gParams.calibrationData, network, gParams.calibBatchSize,
            gParams.calibMaxBatchIdx);
    }
    if (gParams.dataType == DataType::kHALF &&
        !builder->platformHasFastFp16()) {
        LOG_ERROR("The device for FP16 run since its not supported.");
        return false;
    }
    const IBlobNameToTensor *blobNameToTensor = parser->parse(
        gParams.deployFile.c_str(), gParams.modelFile.c_str(), *network,
        gParams.dataType == DataType::kINT8 ? DataType::kFLOAT
                                            : gParams.dataType);
    // specify which tensors are outputs
    for (const auto &output_name : gParams.output_names) {
        if (!blobNameToTensor->find(output_name.c_str())) {
            LOG_ERROR("Could not find output name {}", output_name);
            return false;
        }
        network->markOutput(*blobNameToTensor->find(output_name.c_str()));
    }
    // Build the engine
    IBuilderConfig *config = builder->createBuilderConfig();
    if (gParams.dataType == DataType::kINT8) {
        config->setAvgTimingIterations(gParams.avgRuns);
        config->setMinTimingIterations(gParams.iterations);
        config->setFlag(BuilderFlag::kDEBUG);
        config->setFlag(BuilderFlag::kINT8);
        config->setFlag(BuilderFlag::kFP16);
        config->setInt8Calibrator(calibrator);
    }
    if (gParams.dataType == DataType::kHALF) {
        config->setFlag(BuilderFlag::kFP16);
    }
    if (gParams.DLACore >= 0) {
        assert(gParams.DLACore < builder->getNbDLACores());
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
    config->setMaxWorkspaceSize(static_cast<size_t>(gParams.workspaceSize)
                                << 20);
    builder->setMaxBatchSize(gParams.batchSize);
    ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
    if (!engine) {
        LOG_ERROR("Engine creation failed");
        return false;
    }
    // we don't need the network any more, and we can destroy the parser
    // serialize the engine, then close everything down
    (*trtModelStream) = engine->serialize();
    network->destroy();
    parser->destroy();
    engine->destroy();
    config->destroy();
    builder->destroy();
    shutdownProtobufLibrary();
    return true;
}

int main(int argc, char **argv) {
    InitGoogleLogging();
    if (!parseArgs(argc, argv))
        return -1;

    CUDACHECK(cudaSetDevice(gParams.device));
    if (gParams.output_names.empty()) {
        LOG_ERROR("At least one network output must be defined");
        return -1;
    }

    initLibNvInferPlugins(&gLogger, "");
    nv::Int8EntropyCalibrator *calibrator{nullptr};
    IHostMemory *trtModelStream{nullptr};
    if (!caffeToTrtModel(calibrator, &trtModelStream)) {
        LOG_ERROR("Caffe to trt model failed.");
        return -1;
    }
    std::ofstream trtModelFile(gParams.engine.c_str(), std::ios::binary);
    trtModelFile.write((const char *)(trtModelStream->data()),
                       trtModelStream->size());
    LOG_INFO("Convert model to tensor model cache: {} completed.",
             gParams.engine.c_str());
    trtModelFile.close();
    trtModelStream->destroy();
    SAFE_FREE(calibrator);
    return 0;
}