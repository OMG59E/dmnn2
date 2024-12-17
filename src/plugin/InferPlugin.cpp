/***
 * @Author: xingwg
 * @Date: 2024-10-12 09:59:37
 * @LastEditTime: 2024-12-06 17:25:04
 * @FilePath: /dmnn2/src/plugin/InferPlugin.cpp
 * @Description:
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved.
 */
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <algorithm>
#include <array>
#include <iostream>
#include <memory>

using namespace nvinfer1;
using namespace nvinfer1::plugin;

#include "YOLOBoxPlugin/YOLOBoxPlugin.h"
#include "YOLONMSPlugin/YOLONMSPlugin.h"
#include "YOLONMSPluginV2/YOLONMSPluginV2.h"
#include "YOLOXNMSPlugin/YOLOXNMSPlugin.h"
#include "ctNMSPlugin/ctNMSPlugin.h"
#include "efficientNMSPlugin/efficientNMSPlugin.h"
#include "flatten/flatten.h"
#include "focusPlugin/focusPlugin.h"
#include "instanceNormalizationPlugin/instanceNormalizationPlugin.h"
#include "interpPlugin/interpPlugin.h"
#include "nmsPlugin/nmsPlugin.h"
#include "nmsPluginV2/nmsPluginV2.h"
#include "normalizePlugin/normalizePlugin.h"
#include "priorBoxPlugin/priorBoxPlugin.h"
#include "scaleV2Plugin/scaleV2Plugin.h"
#include "slicePlugin/slicePlugin.h"
#include "upsamplePlugin/upsamplePlugin.h"

namespace nvinfer1::plugin {
ILogger *gLogger{};
template <typename CreatorType> class InitializePlugin {
public:
    InitializePlugin(void *logger, const char *libNamespace)
        : mCreator{new CreatorType{}} {
        mCreator->setPluginNamespace(libNamespace);
        bool status =
            getPluginRegistry()->registerCreator(*mCreator, libNamespace);
        if (logger) {
            nvinfer1::plugin::gLogger =
                static_cast<nvinfer1::ILogger *>(logger);
            if (!status) {
                std::string errorMsg{
                    "Could not register plugin creator:  " +
                    std::string(mCreator->getPluginName()) + " in namespace: " +
                    std::string{mCreator->getPluginNamespace()}};
                nvinfer1::plugin::gLogger->log(ILogger::Severity::kERROR,
                                               errorMsg.c_str());
            } else {
                std::string verboseMsg{
                    "Plugin Creator registration succeeded - " +
                    std::string{mCreator->getPluginName()}};
                nvinfer1::plugin::gLogger->log(ILogger::Severity::kVERBOSE,
                                               verboseMsg.c_str());
            }
        }
    }
    InitializePlugin(const InitializePlugin &) = delete;
    InitializePlugin(InitializePlugin &&) = delete;

private:
    std::unique_ptr<CreatorType> mCreator;
};
template <typename CreatorType>
void initializePlugin(void *logger, const char *libNamespace) {
    static InitializePlugin<CreatorType> plugin{logger, libNamespace};
}
}  // namespace nvinfer1::plugin

extern "C" {
bool initLibNvInferPlugins(void *logger, const char *libNamespace) {
    initializePlugin<nvinfer1::plugin::NMSPluginCreator>(logger, libNamespace);
    initializePlugin<nvinfer1::plugin::NMSPluginV2Creator>(logger,
                                                           libNamespace);
    initializePlugin<nvinfer1::plugin::PriorBoxPluginCreator>(logger,
                                                              libNamespace);
    initializePlugin<nvinfer1::plugin::NormalizePluginCreator>(logger,
                                                               libNamespace);
    // initializePlugin<nvinfer1::plugin::FlattenPluginCreator>(logger,
    // libNamespace);
    // initializePlugin<nvinfer1::plugin::InterpPluginCreator>(logger,
    // libNamespace);
    initializePlugin<nvinfer1::plugin::SlicePluginCreator>(logger,
                                                           libNamespace);
    initializePlugin<nvinfer1::plugin::ScaleV2PluginCreator>(logger,
                                                             libNamespace);
    // initializePlugin<nvinfer1::plugin::UpsamplePluginCreator>(logger,
    // libNamespace);
    initializePlugin<nvinfer1::plugin::YOLOBoxPluginCreator>(logger,
                                                             libNamespace);
    initializePlugin<nvinfer1::plugin::YOLONMSPluginCreator>(logger,
                                                             libNamespace);
    initializePlugin<nvinfer1::plugin::YOLONMSPluginV2Creator>(logger,
                                                               libNamespace);
    initializePlugin<nvinfer1::plugin::YOLONMSDynamicPluginCreator>(
        logger, libNamespace);
    initializePlugin<nvinfer1::plugin::YOLONMSDynamicPluginV2Creator>(
        logger, libNamespace);
    initializePlugin<nvinfer1::plugin::CenterNMSPluginCreator>(logger,
                                                               libNamespace);
    initializePlugin<nvinfer1::plugin::YOLOXNMSPluginCreator>(logger,
                                                              libNamespace);
    initializePlugin<nvinfer1::plugin::FocusPluginCreator>(logger,
                                                           libNamespace);
    initializePlugin<nvinfer1::plugin::InstanceNormalizationPluginCreator>(
        logger, libNamespace);
    initializePlugin<nvinfer1::plugin::EfficientNMSONNXPluginCreator>(
        logger, libNamespace);
    initializePlugin<nvinfer1::plugin::EfficientNMSPluginCreator>(logger,
                                                                  libNamespace);
    return true;
}
}  // extern "C"
