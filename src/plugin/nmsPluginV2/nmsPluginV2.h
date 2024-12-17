/*** 
 * @Author: xingwg
 * @Date: 2024-10-12 09:59:37
 * @LastEditTime: 2024-10-12 11:38:20
 * @FilePath: /dmnn2/src/plugin/nmsPluginV2/nmsPluginV2.h
 * @Description: 
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved. 
 */
#ifndef ALGORITHMS_NMS_LKPLUGIN_H
#define ALGORITHMS_NMS_LKPLUGIN_H
#include <string>
#include <vector>
#include "../common/kernels/kernel.h"
#include "../common/nmsUtils.h"
#include "../common/plugin.h"
#include "error_check.h"

using namespace nvinfer1;
using namespace nvinfer1::plugin;

namespace nvinfer1 {
    namespace plugin {
        struct DetectionOutputV2Parammeters {
            bool shareLocation, varianceEncodedInTarget;
            int backgroundLabelId, numClasses, numPts, topK, keepTopK;
            float confidenceThreshold, nmsThreshold;
            CodeTypeSSD codeType;
            int inputOrder[4] = {0};
            bool confSigmoid;
            bool isNormalized;
        };
        static const char* NMS_PLUGIN_V2_VERSION{"1"};
        static const char* NMS_PLUGIN_V2_NAME{"NMS_V2_TRT"};
        class DetectionOutputV2 : public IPluginV2Ext {
        public:
            explicit DetectionOutputV2(DetectionOutputV2Parammeters param) : param(param) {}
            DetectionOutputV2(DetectionOutputV2Parammeters param, int C1, int C2, int C3, int numPriors)
                    : param(param), C1(C1), C2(C2), C3(C3), numPriors(numPriors) {}
            DetectionOutputV2(const void *data, size_t length) {
                const char *d = reinterpret_cast<const char*>(data), *a = d;
                param = read<DetectionOutputV2Parammeters>(d);
                // Channel size of the locData tensor
                // numPriors * numLocClasses * 4
                C1 = read<int>(d);
                // Channel size of the confData tensor
                // numPriors * param.numClasses
                C2 = read<int>(d);
                //num_priors * param.numPts
                C3 = read<int>(d);
                // Number of bounding boxes per sample
                numPriors = read<int>(d);
                assert(d == a + length);
            }
            ~DetectionOutputV2() override = default;
            int getNbOutputs() const noexcept override { return 1; }
            Dims getOutputDimensions(int index, const Dims *inputs, int nbInputDims) noexcept override {
                assert(nbInputDims == 4);
                assert(index == 0);
                //id score label x1, y1, x2, y2, (pt_x,pt_y)*num_pts
                return Dims3(1, param.keepTopK, 7 + 2*param.numPts);
            }
            int initialize() noexcept override { return STATUS_SUCCESS; }
            void terminate() noexcept override {}
            size_t getWorkspaceSize(int maxBatchSize) const noexcept override {
                return detectionInferenceWorkspaceSize(param.shareLocation, maxBatchSize, C1, C2, C3, param.numClasses, numPriors,
                                                       param.numPts, param.topK, DataType::kFLOAT, DataType::kFLOAT, DataType::kFLOAT);
            }
            int enqueue(int batchSize, void const* const* inputs,
                    void* const* outputs, void *workspace, cudaStream_t stream) noexcept override {
                // Input order {loc, conf, prior,landmark}
                const void* const locData = inputs[param.inputOrder[0]];   // 目标框数据
                const void* const confData = inputs[param.inputOrder[1]];  // 目标类别数据
                const void* const priorData = inputs[param.inputOrder[2]]; // 先验锚点数据
                const void* const landData = inputs[param.inputOrder[3]];  // 关键点数据

                // Output from plugin index 0: topDetections index 1: keepCount
                void* topLKDetections = outputs[0];
                void* keepCount = nullptr;
                pluginStatus_t status = detectionInferenceV2(stream, batchSize, C1, C2, C3, param.shareLocation,
                                                             param.varianceEncodedInTarget, param.backgroundLabelId, numPriors,
                                                             param.numClasses, param.numPts, param.topK, param.keepTopK,
                                                             param.confidenceThreshold, param.nmsThreshold, param.codeType,
                                                             DataType::kFLOAT, locData, priorData, landData,
                                                             DataType::kFLOAT, confData, keepCount, topLKDetections, workspace,
                                                             param.isNormalized, param.confSigmoid);
                assert(status == STATUS_SUCCESS);
                return 0;
            }
            size_t getSerializationSize() const noexcept override { return sizeof(DetectionOutputV2Parammeters) + sizeof(int) * 4; }
            void serialize(void *buffer) const noexcept override {
                char *d = reinterpret_cast<char*>(buffer), *a = d;
                write(d, param);
                write(d, C1);
                write(d, C2);
                write(d, C3);
                write(d, numPriors);
                assert(d == a + getSerializationSize());
            }
            bool supportsFormat(DataType type, PluginFormat format) const noexcept override { return (type == DataType::kFLOAT && format == PluginFormat::kLINEAR); }
            const char *getPluginType() const noexcept override { return NMS_PLUGIN_V2_NAME; }
            const char *getPluginVersion() const noexcept override { return NMS_PLUGIN_V2_VERSION; }
            void destroy() noexcept override { delete this; }
            IPluginV2Ext *clone() const noexcept override {
                auto* plugin = new DetectionOutputV2(param, C1, C2, C3, numPriors);
                plugin->setPluginNamespace(mPluginNamespace);
                return plugin;
            }
            void setPluginNamespace(const char *pluginNamespace) noexcept override { mPluginNamespace = pluginNamespace; }
            const char *getPluginNamespace() const noexcept override { return mPluginNamespace; }
            DataType getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const noexcept override {
                assert(index == 0);
                return DataType::kFLOAT;
            }
            bool isOutputBroadcastAcrossBatch(int outputIndex, const bool *inputIsBroadcasted, int nbInputs) const noexcept override { return false; }
            bool canBroadcastInputAcrossBatch(int inputIndex) const noexcept override { return false; }
            void attachToContext(cudnnContext *cudnnContext, cublasContext *cublasContext, IGpuAllocator *gpuAllocator) noexcept override {}
            void configurePlugin(const Dims *inputDims, int nbInputs, const Dims *outputDims, int nbOutputs,
                                 const DataType *inputTypes, const DataType *outputTypes, const bool *inputIsBroadcast,
                                 const bool *outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) noexcept override {
                assert(nbInputs == 4);
                assert(nbOutputs == 1);
                for (int i = 0; i < nbInputs; i++)
                    assert(inputDims[i].nbDims == 3);
                for (int i = 0; i < nbOutputs; i++)
                    assert(outputDims[i].nbDims == 3);
                C1 = inputDims[param.inputOrder[0]].d[0];  // 先验框的数量 * 框的位置(x1, y1, x2, y2)
                C2 = inputDims[param.inputOrder[1]].d[0];  // 先验框的数量 * 框的类别
                C3 = inputDims[param.inputOrder[3]].d[0];  // 先验框的数量 * 2 * num_pts
                const int nbBoxCoordinates = 4;
                numPriors = inputDims[param.inputOrder[2]].d[1] / nbBoxCoordinates;
                const int numLocClasses = param.shareLocation ? 1 : param.numClasses;
                assert(numPriors * numLocClasses * nbBoxCoordinates == C1);
                assert(numPriors * param.numClasses == C2);
                assert(numPriors * numLocClasses * 2 * param.numPts == C3);
            }
            void detachFromContext() noexcept override {}
        private:
            DetectionOutputV2Parammeters param{};
            int C1{0}, C2{0}, C3{0}, numPriors{0};
            const char *mPluginNamespace{""};
        };

        class NMSPluginV2Creator : public BaseCreator {
        public:
            NMSPluginV2Creator() {
                // NMS Plugin field meta data {name,  data, type, length}
                mPluginAttributes.clear();
                mPluginAttributes.emplace_back(PluginField("shareLocation", nullptr, PluginFieldType::kINT32, 1));
                mPluginAttributes.emplace_back(PluginField("varianceEncodedInTarget", nullptr, PluginFieldType::kINT32, 1));
                mPluginAttributes.emplace_back(PluginField("backgroundLabelId", nullptr, PluginFieldType::kINT32, 1));
                mPluginAttributes.emplace_back(PluginField("numClasses", nullptr, PluginFieldType::kINT32, 1));
                mPluginAttributes.emplace_back(PluginField("topK", nullptr, PluginFieldType::kINT32, 1));
                mPluginAttributes.emplace_back(PluginField("keepTopK", nullptr, PluginFieldType::kINT32, 1));
                mPluginAttributes.emplace_back(PluginField("confidenceThreshold", nullptr, PluginFieldType::kFLOAT32, 1));
                mPluginAttributes.emplace_back(PluginField("nmsThreshold", nullptr, PluginFieldType::kFLOAT32, 1));
                mPluginAttributes.emplace_back(PluginField("inputOrder", nullptr, PluginFieldType::kINT32, 3));
                mPluginAttributes.emplace_back(PluginField("confSigmoid", nullptr, PluginFieldType::kINT32, 1));
                mPluginAttributes.emplace_back(PluginField("isNormalized", nullptr, PluginFieldType::kINT32, 1));
                mPluginAttributes.emplace_back(PluginField("codeType", nullptr, PluginFieldType::kINT32, 1));
                mPluginAttributes.emplace_back(PluginField("numPts", nullptr, PluginFieldType::kINT32, 1));
                mFC.nbFields = mPluginAttributes.size();
                mFC.fields = mPluginAttributes.data();
            }
            ~NMSPluginV2Creator() override = default;
            const char *getPluginName() const noexcept override { return NMS_PLUGIN_V2_NAME; }
            const char *getPluginVersion() const noexcept override { return NMS_PLUGIN_V2_VERSION; }
            const PluginFieldCollection *getFieldNames() noexcept override { return &mFC; }
            IPluginV2Ext *createPlugin(const char *name, const PluginFieldCollection *fc) noexcept override {
                DetectionOutputV2Parammeters params{};
                const PluginField* fields = fc->fields;
                // Default init values for TF SSD network
                params.codeType = CodeTypeSSD::TF_CENTER;
                params.inputOrder[0] = 0;
                params.inputOrder[1] = 2;
                params.inputOrder[2] = 1;
                params.inputOrder[3] = 3;
                // Read configurations from  each fields
                for (int i = 0; i < fc->nbFields; ++i) {
                    const char* attrName = fields[i].name;
                    if (!strcmp(attrName, "shareLocation")) {
                        assert(fields[i].type == PluginFieldType::kINT32);
                        params.shareLocation = static_cast<bool>(*(static_cast<const int*>(fields[i].data)));
                    } else if (!strcmp(attrName, "varianceEncodedInTarget")) {
                        assert(fields[i].type == PluginFieldType::kINT32);
                        params.varianceEncodedInTarget = static_cast<bool>(*(static_cast<const int*>(fields[i].data)));
                    } else if (!strcmp(attrName, "backgroundLabelId")) {
                        assert(fields[i].type == PluginFieldType::kINT32);
                        params.backgroundLabelId = *(static_cast<const int*>(fields[i].data));
                    } else if (!strcmp(attrName, "numClasses")) {
                        assert(fields[i].type == PluginFieldType::kINT32);
                        params.numClasses = *(static_cast<const int*>(fields[i].data));
                    } else if (!strcmp(attrName, "topK")) {
                        assert(fields[i].type == PluginFieldType::kINT32);
                        params.topK = *(static_cast<const int*>(fields[i].data));
                    } else if (!strcmp(attrName, "keepTopK")) {
                        assert(fields[i].type == PluginFieldType::kINT32);
                        params.keepTopK = *(static_cast<const int*>(fields[i].data));
                    } else if (!strcmp(attrName, "confidenceThreshold")) {
                        assert(fields[i].type == PluginFieldType::kFLOAT32);
                        params.confidenceThreshold = *(static_cast<const float*>(fields[i].data));
                    } else if (!strcmp(attrName, "nmsThreshold")) {
                        assert(fields[i].type == PluginFieldType::kFLOAT32);
                        params.nmsThreshold = *(static_cast<const float*>(fields[i].data));
                    } else if (!strcmp(attrName, "confSigmoid")) {
                        params.confSigmoid = static_cast<bool>(*(static_cast<const int*>(fields[i].data)));
                    } else if (!strcmp(attrName, "isNormalized")) {
                        params.isNormalized = static_cast<bool>(*(static_cast<const int*>(fields[i].data)));
                    } else if (!strcmp(attrName, "inputOrder")) {
                        assert(fields[i].type == PluginFieldType::kINT32);
                        const int size = fields[i].length;
                        const int* o = static_cast<const int*>(fields[i].data);
                        for (int j = 0; j < size; j++) {
                            params.inputOrder[j] = *o;
                            o++;
                        }
                    } else if (!strcmp(attrName, "codeType")) {
                        assert(fields[i].type == PluginFieldType::kINT32);
                        params.codeType = static_cast<CodeTypeSSD>(*(static_cast<const int*>(fields[i].data)));
                    } else if (!strcmp(attrName, "numPts")) {
                        assert(fields[i].type == PluginFieldType::kINT32);
                        params.numPts = *(static_cast<const int*>(fields[i].data));
                    }
                }
                auto* plugin = new DetectionOutputV2(params);
                plugin->setPluginNamespace(mNamespace.c_str());
                return plugin;
            }

            IPluginV2Ext *deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept override {
                auto* plugin = new DetectionOutputV2(serialData, serialLength);
                plugin->setPluginNamespace(mNamespace.c_str());
                return plugin;
            }

        private:
            PluginFieldCollection mFC{};
            std::vector<PluginField> mPluginAttributes{};
        };
    }
}

#endif //ALGORITHMS_NMS_LKPLUGIN_H
