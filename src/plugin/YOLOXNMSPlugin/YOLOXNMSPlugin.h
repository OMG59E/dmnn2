/*** 
 * @Author: xingwg
 * @Date: 2024-10-12 09:59:37
 * @LastEditTime: 2024-10-12 13:57:35
 * @FilePath: /dmnn2/src/plugin/YOLOXNMSPlugin/YOLOXNMSPlugin.h
 * @Description: 
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved. 
 */

#ifndef ALGORITHMS_YOLOXNMSPLUGIN_H
#define ALGORITHMS_YOLOXNMSPLUGIN_H

#include <string>
#include <vector>
#include "../common/kernels/kernel.h"
#include "../common/nmsUtils.h"
#include "../common/plugin.h"
#include "error_check.h"
#include "base_types.h"

struct YOLOXDetectionOutputParameters {
    int num_classes{4};
    float nms_thresh{0.45f};
    float conf_thresh{0.01f};
    int keep_topK{200};
    int topK{400};
    bool use_p6{false};
};

using namespace nvinfer1;
using namespace nvinfer1::plugin;

namespace nvinfer1 {
    namespace plugin {
        static const char *YOLOX_NMS_PLUGIN_VERSION{"1"};
        static const char *YOLOX_NMS_PLUGIN_NAME{"YOLOX_NMS_TRT"};
        class YOLOXDetectionOutput : public IPluginV2Ext {
        public:
            explicit YOLOXDetectionOutput(YOLOXDetectionOutputParameters param)
                    : param(param) {}
            YOLOXDetectionOutput(YOLOXDetectionOutputParameters param, int C1, int C2)
                    : param(param) , C1(C1) , C2(C2) {}
            YOLOXDetectionOutput(const void *data, size_t length) {
                const char *d = reinterpret_cast<const char *>(data);
                param = read<YOLOXDetectionOutputParameters>(d);
                C1 = read<int>(d);
                C2 = read<int>(d);
                assert(d == data + length);
            }
            ~YOLOXDetectionOutput() override = default;

            int getNbOutputs() const noexcept override { return 1; }
            Dims getOutputDimensions(int index, const Dims *inputs, int nbInputDims) noexcept override {
                assert(nbInputDims == 1);
                assert(index == 0);
                return Dims3(1, param.keep_topK, 7);
            }

            int initialize() noexcept override { return STATUS_SUCCESS; }
            void terminate() noexcept override {}
            size_t getWorkspaceSize(int maxBatchSize) const noexcept override {
                return detectionInferenceWorkspaceSize(maxBatchSize,
                        C1, C2, param.num_classes, param.topK, DataType::kFLOAT, DataType::kFLOAT);
            }

            int enqueue(int batchSize, void const*const *inputs, void *const* outputs, void *workspace, cudaStream_t stream) noexcept override;
            size_t getSerializationSize() const noexcept override { return sizeof(YOLOXDetectionOutputParameters) + sizeof(int) * 2; }
            void serialize(void *buffer) const noexcept override {
                char *d = reinterpret_cast<char *>(buffer);
                write(d, param);
                write(d, C1);
                write(d, C2);
                assert(d == buffer + getSerializationSize());
            }

            bool supportsFormat(DataType type, PluginFormat format) const noexcept override { return (type == DataType::kFLOAT && format == PluginFormat::kLINEAR); }
            const char *getPluginType() const noexcept override { return YOLOX_NMS_PLUGIN_NAME; }
            const char *getPluginVersion() const noexcept override { return YOLOX_NMS_PLUGIN_VERSION; }
            void destroy() noexcept override { delete this; }
            IPluginV2Ext *clone() const noexcept override {
                auto *plugin = new YOLOXDetectionOutput(param, C1, C2);
                plugin->setPluginNamespace(mPluginNamespace.c_str());
                return plugin;
            }

            void setPluginNamespace(const char *pluginNamespace) noexcept override { mPluginNamespace = pluginNamespace; }
            const char *getPluginNamespace() const noexcept override { return mPluginNamespace.c_str(); }
            DataType getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const noexcept override { return DataType::kFLOAT; }

            bool isOutputBroadcastAcrossBatch(int outputIndex, const bool *inputIsBroadcasted, int nbInputs) const noexcept override { return false; }
            bool canBroadcastInputAcrossBatch(int inputIndex) const noexcept override { return false; }
            void attachToContext(cudnnContext *cudnnContext, cublasContext *cublasContext, IGpuAllocator *gpuAllocator) noexcept override {}
            void configurePlugin(const Dims *inputDims, int nbInputs, const Dims *outputDims, int nbOutputs,
                                 const DataType *inputTypes, const DataType *outputTypes, const bool *inputIsBroadcast,
                                 const bool *outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) noexcept override {
                assert(nbInputs == 1);
                assert(nbOutputs == 1);
                for (int i = 0; i < nbInputs; i++)
                    assert(inputDims[i].nbDims == 3);
                for (int i = 0; i < nbOutputs; i++)
                    assert(outputDims[i].nbDims == 3);

                C1 = inputDims[0].d[0]; // 52*52 + 26*26 + 13*13
                C2 = inputDims[0].d[1]; // 4 + 1 + cls
            }

            void detachFromContext() noexcept override {}

        private:
            YOLOXDetectionOutputParameters param{};
            int C1{0}, C2{0};
            std::string mPluginNamespace;
        };

        class YOLOXNMSPluginCreator : public BaseCreator {
        public:
            YOLOXNMSPluginCreator() {
                // NMS Plugin field meta data {name,  data, type, length}
                mPluginAttributes.clear();
                mPluginAttributes.emplace_back(PluginField("conf_thresh", nullptr, PluginFieldType::kFLOAT32, 1));
                mPluginAttributes.emplace_back(PluginField("nms_thresh", nullptr, PluginFieldType::kFLOAT32, 1));
                mPluginAttributes.emplace_back(PluginField("keep_topK", nullptr, PluginFieldType::kINT32, 1));
                mPluginAttributes.emplace_back(PluginField("topK", nullptr, PluginFieldType::kINT32, 1));
                mPluginAttributes.emplace_back(PluginField("num_classes", nullptr, PluginFieldType::kINT32, 1));
                mPluginAttributes.emplace_back(PluginField("use_p6", nullptr, PluginFieldType::kINT32, 1));
                mFC.nbFields = static_cast<int32_t>(mPluginAttributes.size());
                mFC.fields = mPluginAttributes.data();
            }
            ~YOLOXNMSPluginCreator() override = default;
            const char *getPluginName() const noexcept override { return YOLOX_NMS_PLUGIN_NAME; }
            const char *getPluginVersion() const noexcept override { return YOLOX_NMS_PLUGIN_VERSION; }
            const PluginFieldCollection *getFieldNames() noexcept override { return &mFC; }
            IPluginV2Ext *createPlugin(const char *name, const PluginFieldCollection *fc) noexcept override {
                YOLOXDetectionOutputParameters params{};
                const PluginField *fields = fc->fields;
                for (int i = 0; i < fc->nbFields; ++i) {
                    const char *attrName = fields[i].name;
                    if (!strcmp(attrName, "conf_thresh")) {
                        assert(fields[i].type == PluginFieldType::kFLOAT32);
                        params.conf_thresh = *(static_cast<const float *>(fields[i].data));
                    } else if (!strcmp(attrName, "nms_thresh")) {
                        assert(fields[i].type == PluginFieldType::kFLOAT32);
                        params.nms_thresh = *(static_cast<const float *>(fields[i].data));
                    } else if (!strcmp(attrName, "keep_topK")) {
                        assert(fields[i].type == PluginFieldType::kINT32);
                        params.keep_topK = *(static_cast<const int *>(fields[i].data));
                    } else if (!strcmp(attrName, "topK")) {
                        assert(fields[i].type == PluginFieldType::kINT32);
                        params.topK = *(static_cast<const int *>(fields[i].data));
                    } else if (!strcmp(attrName, "num_classes")) {
                        assert(fields[i].type == PluginFieldType::kINT32);
                        params.num_classes = *(static_cast<const int *>(fields[i].data));
                    } else if (!strcmp(attrName, "use_p6")) {
                        assert(fields[i].type == PluginFieldType::kINT32);
                        params.use_p6 = (bool)(*(static_cast<const int *>(fields[i].data)));
                    }
                }
                auto *plugin = new YOLOXDetectionOutput(params);
                plugin->setPluginNamespace(mNamespace.c_str());
                return plugin;
            }

            IPluginV2Ext *deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept override {
                auto *plugin = new YOLOXDetectionOutput(serialData, serialLength);
                plugin->setPluginNamespace(mNamespace.c_str());
                return plugin;
            }

        private:
            PluginFieldCollection mFC{};
            std::vector<PluginField> mPluginAttributes{};
        };
    } // namespace plugin
} // namespace nvinfer1

#endif //ALGORITHMS_YOLOXNMSPLUGIN_H
