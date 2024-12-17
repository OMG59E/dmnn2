/***
 * @Author: xingwg
 * @Date: 2024-10-12 21:31:59
 * @LastEditTime: 2024-10-16 11:13:23
 * @FilePath: /dmnn2/src/utils/trt_logger.h
 * @Description:
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#pragma once
#include "logging.h"
#include <NvInfer.h>
#include <iostream>

using namespace nvinfer1;

class TrtLogger : public ILogger {
public:
    explicit TrtLogger(Severity severity = Severity::kWARNING)
        : reportableSeverity(severity) {}

    void log(Severity severity, const char *msg) noexcept override {
        // suppress messages with severity enum value greater than the
        // reportable
        if (severity > reportableSeverity)
            return;

        switch (severity) {
        case Severity::kINTERNAL_ERROR:
            LOG_ERROR("INTERNAL: {}", msg);
            break;
        case Severity::kERROR:
            LOG_ERROR(msg);
            break;
        case Severity::kWARNING:
            LOG_WARNING(msg);
            break;
        case Severity::kINFO:
            LOG_INFO(msg);
            break;
        case Severity::kVERBOSE:
            LOG_INFO(msg);
            break;
        }
    }
    Severity reportableSeverity;
};

struct TrtProfiler : public IProfiler {
    typedef std::pair<std::string, float> Record;
    std::vector<Record> mProfiles;
    void reportLayerTime(const char *layerName, float ms) noexcept override {
        auto record =
            std::find_if(mProfiles.begin(), mProfiles.end(),
                         [&](const Record &r) { return r.first == layerName; });
        if (record == mProfiles.end()) {
            mProfiles.emplace_back(std::make_pair(layerName, ms));
        } else {
            record->second += ms;
        }
    }
    void printLayerTimes(const int iterations) {
        float totalTime = 0;
        for (auto &profile : mProfiles) {
            LOG_INFO("{}: {:.6f}ms", profile.first,
                     profile.second / iterations);
            totalTime += profile.second;
        }
        LOG_INFO("Time over all layers: {:.6f}ms", totalTime / iterations);
    }
};
