/*** 
 * @Author: xingwg
 * @Date: 2024-10-12 09:59:37
 * @LastEditTime: 2024-10-12 14:07:49
 * @FilePath: /dmnn2/src/plugin/ctNMSPlugin/ctNMSPlugin.h
 * @Description: 
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved. 
 */
#ifndef ALGORITHMS_CENTERNMSPLUGIN_H
#define ALGORITHMS_CENTERNMSPLUGIN_H
#include <string>
#include <vector>
#include <cassert>
#include "../common/kernels/kernel.h"
#include "../common/nmsUtils.h"
#include "../common/plugin.h"
#include "error_check.h"
#include "base_types.h"

using namespace nvinfer1::plugin;

struct CenterFaceOutputParameters {
    int num_pts;
    int num_classes;
    int topK;
    int keep_topK;
    float nms_threshold;
    float confidence_threshold;
};

namespace nvinfer1 {
    namespace plugin {
        static const char *CENTER_NMS_PLUGIN_VERSION{ "1" };
        static const char *CENTER_NMS_PLUGIN_NAME{ "CT_NMS_TRT" };
        class CenterFaceOutput : public IPluginV2Ext {
        public:
            explicit CenterFaceOutput(CenterFaceOutputParameters param) : param_(param) {};
            CenterFaceOutput(CenterFaceOutputParameters param, int H, int W) : param_(param), H_(H), W_(W) {}
            CenterFaceOutput(const void *data, size_t length) {
                const char *d = reinterpret_cast<const char *>(data);
                param_ = read<CenterFaceOutputParameters>(d);
                H_ = read<int>(d);
                W_ = read<int>(d);
                assert(d == data + length);
            }
            ~CenterFaceOutput() override = default;
            int getNbOutputs() const noexcept override { return 1; }
            Dims getOutputDimensions(int index, const Dims *inputs, int nbInputDims) noexcept override {
                assert(nbInputDims == 4);
                assert(index == 0);
                return Dims3(1, param_.keep_topK, 7 + 2 * param_.num_pts);
            }
            int initialize() noexcept override { return STATUS_SUCCESS; }
            void terminate() noexcept override {}
            size_t getWorkspaceSize(int maxBatchSize) const noexcept override {
                // N*1*(HW)*15
                size_t wss[7];
                wss[0] = maxBatchSize*H_*W_*4*sizeof(float);
                wss[1] = maxBatchSize*H_*W_*1*sizeof(float);
                wss[2] = maxBatchSize*H_*W_*2*param_.num_pts*sizeof(float);
                wss[3] = detectionForwardPreNMSSize(maxBatchSize, H_*W_*param_.num_classes);
                wss[4] = detectionForwardPostNMSSize(maxBatchSize, param_.num_classes, param_.topK);
                wss[5] = detectionForwardPostNMSSize(maxBatchSize, param_.num_classes, param_.topK);
                wss[6] = std::max(sortScoresPerClassWorkspaceSize(maxBatchSize, param_.num_classes, H_*W_, DataType::kFLOAT),
                                  sortScoresPerImageWorkspaceSize(maxBatchSize, param_.num_classes * param_.topK, DataType::kFLOAT));
                return calculateTotalWorkspaceSize(wss, 7);
            }
            int enqueue(int32_t batchSize, void const* const* inputs, void* const* outputs, void *workspace, cudaStream_t stream) noexcept override;
            size_t getSerializationSize() const noexcept override { return sizeof(CenterFaceOutputParameters) + sizeof(int) * 2; }
            void serialize(void *buffer) const noexcept override {
                char *d = reinterpret_cast<char *>(buffer);
                write(d, param_);
                write(d, H_);
                write(d, W_);
                assert(d == buffer + getSerializationSize());
            }
            bool supportsFormat(DataType type, PluginFormat format) const noexcept override { return (type == DataType::kFLOAT && format == PluginFormat::kLINEAR); }
            const char *getPluginType() const noexcept override { return CENTER_NMS_PLUGIN_NAME; }
            const char *getPluginVersion() const noexcept override { return CENTER_NMS_PLUGIN_VERSION; }
            void destroy() noexcept override { delete this; }
            IPluginV2Ext *clone() const noexcept override {
                auto *plugin = new CenterFaceOutput(param_, H_, W_);
                plugin->setPluginNamespace(mPluginNamespace_.c_str());
                return plugin;
            }
            void setPluginNamespace(const char *pluginNamespace) noexcept override { mPluginNamespace_ = pluginNamespace; }
            const char *getPluginNamespace() const noexcept override { return mPluginNamespace_.c_str(); }
            DataType getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const noexcept override { return DataType::kFLOAT; }
            bool isOutputBroadcastAcrossBatch(int outputIndex, const bool *inputIsBroadcasted, int nbInputs) const noexcept override { return false; }
            bool canBroadcastInputAcrossBatch(int inputIndex) const noexcept override { return false; }
            void attachToContext(cudnnContext *cudnnContext, cublasContext *cublasContext, IGpuAllocator *gpuAllocator) noexcept override {}
            void configurePlugin(const Dims *inputDims, int nbInputs, const Dims *outputDims, int nbOutputs,
                                 const DataType *inputTypes, const DataType *outputTypes, const bool *inputIsBroadcast,
                                 const bool *outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) noexcept override {
                assert(nbInputs == 4 && nbOutputs == 1);
                for (int i = 0; i < nbInputs; i++)
                    assert(inputDims[i].nbDims == 3);
                assert(inputDims[0].d[0] == param_.num_classes
                           && inputDims[1].d[0] == 2
                           && inputDims[2].d[0] == 2
                           && inputDims[3].d[0] == 2 * param_.num_pts);

                H_ = inputDims[0].d[1];
                W_ = inputDims[0].d[2];

                assert(inputDims[1].d[1] == H_ && inputDims[2].d[1] == H_ && inputDims[3].d[2] == H_);
                assert(inputDims[1].d[2] == W_ && inputDims[2].d[2] == W_ && inputDims[3].d[2] == W_);

                for (int i = 0; i < nbOutputs; i++)
                    assert(outputDims[i].nbDims == 3);
            }
            void detachFromContext() noexcept override {}

        private:
            pluginStatus_t ctFaceNMSInference(cudaStream_t stream, int batch, const float *hmData,
                                              const float *scaleData, const float *offsetData, const float *landDataRaw,
                                              void *workspace, float *topDetections);
        private:
            CenterFaceOutputParameters param_{};
            int H_{0}, W_{0};
            std::string mPluginNamespace_;
        };

        class CenterNMSPluginCreator : public BaseCreator {
        public:
            CenterNMSPluginCreator() {
                // NMS Plugin field meta data {name,  data, type, length}
                mPluginAttributes.emplace_back(PluginField("num_pts", nullptr, PluginFieldType::kINT32, 1));
                mPluginAttributes.emplace_back(PluginField("num_classes", nullptr, PluginFieldType::kINT32, 1));
                mPluginAttributes.emplace_back(PluginField("topK", nullptr, PluginFieldType::kINT32, 1));
                mPluginAttributes.emplace_back(PluginField("keep_topK", nullptr, PluginFieldType::kINT32, 1));
                mPluginAttributes.emplace_back(PluginField("nms_threshold", nullptr, PluginFieldType::kFLOAT32, 1));
                mPluginAttributes.emplace_back(PluginField("confidence_threshold", nullptr, PluginFieldType::kFLOAT32, 1));
                mFC.nbFields = mPluginAttributes.size();
                mFC.fields = mPluginAttributes.data();
            }
            ~CenterNMSPluginCreator() noexcept override = default;
            const char *getPluginName() const noexcept override { return CENTER_NMS_PLUGIN_NAME; }
            const char *getPluginVersion() const noexcept override { return CENTER_NMS_PLUGIN_VERSION; }
            const PluginFieldCollection *getFieldNames() noexcept override { return &mFC; }
            IPluginV2Ext *createPlugin(const char *name, const PluginFieldCollection *fc) noexcept override {
                const PluginField *fields = fc->fields;
                CenterFaceOutputParameters params{};
                for (int i = 0; i < fc->nbFields; ++i) {
                    const char *attrName = fields[i].name;
                    if (!strcmp(attrName, "num_pts")) {
                        assert(fields[i].type == PluginFieldType::kINT32);
                        params.num_pts = *(static_cast<const int32_t*>(fields[i].data));
                    } else if (!strcmp(attrName, "num_classes")) {
                        assert(fields[i].type == PluginFieldType::kINT32);
                        params.num_classes = *(static_cast<const int32_t *>(fields[i].data));
                    } else if (!strcmp(attrName, "topK")) {
                        assert(fields[i].type == PluginFieldType::kINT32);
                        params.topK = *(static_cast<const int32_t *>(fields[i].data));
                    } else if (!strcmp(attrName, "keep_topK")) {
                        assert(fields[i].type == PluginFieldType::kINT32);
                        params.keep_topK = *(static_cast<const int32_t *>(fields[i].data));
                    } else if (!strcmp(attrName, "nms_threshold")) {
                        assert(fields[i].type == PluginFieldType::kFLOAT32);
                        params.nms_threshold = *(static_cast<const float*>(fields[i].data));
                    } else if (!strcmp(attrName, "confidence_threshold")) {
                        assert(fields[i].type == PluginFieldType::kFLOAT32);
                        params.confidence_threshold = *(static_cast<const float*>(fields[i].data));
                    }
                }
                auto *plugin = new CenterFaceOutput(params);
                plugin->setPluginNamespace(mNamespace.c_str());
                return plugin;
            }

            IPluginV2Ext *deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept override {
                auto *plugin = new CenterFaceOutput(serialData, serialLength);
                plugin->setPluginNamespace(mNamespace.c_str());
                return plugin;
            }

        private:
            PluginFieldCollection mFC{};
            std::vector<PluginField> mPluginAttributes{};
        };
    } // namespace plugin
} // namespace nvinfer1

#endif //ALGORITHMS_CENTERNMSPLUGIN_H
