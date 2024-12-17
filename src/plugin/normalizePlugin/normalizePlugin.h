/*** 
 * @Author: xingwg
 * @Date: 2024-10-12 09:59:37
 * @LastEditTime: 2024-10-12 11:38:47
 * @FilePath: /dmnn2/src/plugin/normalizePlugin/normalizePlugin.h
 * @Description: 
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved. 
 */
#ifndef TRT_NORMALIZE_PLUGIN_H
#define TRT_NORMALIZE_PLUGIN_H
#include <string>
#include <vector>
#include <cudnn.h>
#include <cublas_v2.h>
#include "../common/kernels/kernel.h"
#include "../common/plugin.h"
#include "error_check.h"

namespace nvinfer1 {
    namespace plugin {
        static const char *NORMALIZE_PLUGIN_VERSION{"1"};
        static const char *NORMALIZE_PLUGIN_NAME{"Normalize_TRT"};
        class Normalize : public IPluginV2Ext {
        public:
            Normalize(const Weights *weights, int nbWeights, bool acrossSpatial, bool channelShared, float eps)
                    : acrossSpatial(acrossSpatial), channelShared(channelShared), eps(eps) {
                mNbWeights = nbWeights;
                assert(nbWeights == 1);
                assert(weights[0].count >= 1);
                mWeights = copyToDevice(weights[0].values, weights[0].count);
                CUBLASCHECK(cublasCreate(&mCublas));
            }
            Normalize(const Weights *weights, int nbWeights, bool acrossSpatial,
                    bool channelShared, float eps, int C, int H, int W)
                    : acrossSpatial(acrossSpatial), channelShared(channelShared), eps(eps), C(C), H(H), W(W) {
                mNbWeights = nbWeights;
                assert(nbWeights == 1);
                assert(weights[0].count >= 1);
                mWeights = copyToDevice(weights[0].values, weights[0].count);
                CUBLASCHECK(cublasCreate(&mCublas));
            }
            Normalize(const void *buffer, size_t length) {
                const char *d = reinterpret_cast<const char*>(buffer), *a = d;
                C = read < int > (d);
                H = read < int > (d);
                W = read < int > (d);
                acrossSpatial = read < bool > (d);
                channelShared = read < bool > (d);
                eps = read < float > (d);
                int nbWeights = read < int > (d);
                mWeights = deserializeToDevice(d, nbWeights);
                cublasCreate(&mCublas);
                assert(d == a + length);
            }
            ~Normalize() override = default;
            int getNbOutputs() const noexcept override { return 1; }
            Dims getOutputDimensions(int index, const Dims *inputs, int nbInputDims) noexcept override {
                assert(nbInputDims == 1);
                assert(index == 0);
                assert(inputs[0].nbDims == 3);
                return Dims3(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]);
            }
            int initialize() noexcept override { return STATUS_SUCCESS; }
            void terminate() noexcept override { CUBLASCHECK(cublasDestroy(mCublas)); }
            size_t getWorkspaceSize(int maxBatchSize) const noexcept override {
                return normalizePluginWorkspaceSize(acrossSpatial, C, H, W);
            }
            int enqueue(int batchSize, void const* const *inputs, void* const* outputs, void *workspace,
                    cudaStream_t stream) noexcept override {
                const void *inputData = inputs[0];
                void *outputData = outputs[0];
                pluginStatus_t status = normalizeInference(stream, mCublas, acrossSpatial, channelShared, batchSize, C, H, W, eps,
                                                           reinterpret_cast<const float *>(mWeights.values), inputData, outputData,
                                                           workspace);
                assert(status == STATUS_SUCCESS);
                return 0;
            }
            size_t getSerializationSize() const noexcept override { return sizeof(int) * 3 + sizeof(bool) * 2 + sizeof(float) + sizeof(int) + mWeights.count * sizeof(float); }
            void serialize(void *buffer) const noexcept override {
                char *d = reinterpret_cast<char *>(buffer), *a = d;
                write(d, C);
                write(d, H);
                write(d, W);
                write(d, acrossSpatial);
                write(d, channelShared);
                write(d, eps);
                write(d, (int) mWeights.count);
                serializeFromDevice(d, mWeights);
                assert(d == a + getSerializationSize());
            }
            bool supportsFormat(DataType type, PluginFormat format) const noexcept override {
                return (type == DataType::kFLOAT && format == PluginFormat::kLINEAR);
            }
            const char *getPluginType() const noexcept override { return NORMALIZE_PLUGIN_NAME; }
            const char *getPluginVersion() const noexcept override { return NORMALIZE_PLUGIN_VERSION; }
            void destroy() noexcept override { delete this; }
            IPluginV2Ext *clone() const noexcept override {
                auto *plugin = new Normalize(&mWeights, 1, acrossSpatial, channelShared, eps, C, H, W);
                plugin->setPluginNamespace(mPluginNamespace);
                return plugin;
            }
            void setPluginNamespace(const char *pluginNamespace) noexcept override { mPluginNamespace = pluginNamespace; }
            const char *getPluginNamespace() const noexcept override { return mPluginNamespace; }
            DataType getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const noexcept override { return DataType::kFLOAT; }
            bool isOutputBroadcastAcrossBatch(int outputIndex, const bool *inputIsBroadcasted, int nbInputs) const noexcept override { return false; }
            bool canBroadcastInputAcrossBatch(int inputIndex) const noexcept override { return false; }
            void attachToContext(cudnnContext *cudnnContext, cublasContext *cublasContext, IGpuAllocator *gpuAllocator) noexcept override {}
            void configurePlugin(const Dims *inputDims, int nbInputs, const Dims *outputDims, int nbOutputs,
                                 const DataType *inputTypes, const DataType *outputTypes, const bool *inputIsBroadcast,
                                 const bool *outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) noexcept override {
                assert(*inputTypes == DataType::kFLOAT && floatFormat == PluginFormat::kLINEAR);
                C = inputDims[0].d[0];
                H = inputDims[0].d[1];
                W = inputDims[0].d[2];
                if (channelShared) {
                    assert(mWeights.count == 1);
                } else {
                    assert(mWeights.count == C);
                }
                assert(nbInputs == 1);
                assert(nbOutputs == 1);
                assert(inputDims[0].nbDims >= 1); // number of dimensions of the input tensor must be >=2
                assert(inputDims[0].d[0] == outputDims[0].d[0] && inputDims[0].d[1] == outputDims[0].d[1] && inputDims[0].d[2] == outputDims[0].d[2]);
            }
            void detachFromContext() noexcept override {}
        private:
            Weights copyToDevice(const void *hostData, size_t count) {
                void *deviceData;
                CUDACHECK(cudaMalloc(&deviceData, count * sizeof(float)));
                CUDACHECK(cudaMemcpy(deviceData, hostData, count * sizeof(float), cudaMemcpyHostToDevice));
                return Weights{DataType::kFLOAT, deviceData, int64_t(count)};
            }
            void serializeFromDevice(char *&hostBuffer, Weights deviceWeights) const {
                CUDACHECK(cudaMemcpy(hostBuffer, deviceWeights.values, deviceWeights.count * sizeof(float), cudaMemcpyDeviceToHost));
                hostBuffer += deviceWeights.count * sizeof(float);
            }
            Weights deserializeToDevice(const char *&hostBuffer, size_t count) {
                Weights w = copyToDevice(hostBuffer, count);
                hostBuffer += count * sizeof(float);
                return w;
            }

        private:
            cublasHandle_t mCublas{};
            int C{};
            int H{};
            int W{};
            int mNbWeights{};
            bool acrossSpatial{};
            bool channelShared{};
            float eps{};
            Weights mWeights{};
            const char *mPluginNamespace;
        };

        class NormalizePluginCreator : public BaseCreator {
        public:
            NormalizePluginCreator() {
                mPluginAttributes.clear();
                mPluginAttributes.emplace_back(PluginField("weights", nullptr, PluginFieldType::kFLOAT32, 1));
                mPluginAttributes.emplace_back(PluginField("acrossSpatial", nullptr, PluginFieldType::kINT32, 1));
                mPluginAttributes.emplace_back(PluginField("channelShared", nullptr, PluginFieldType::kINT32, 1));
                mPluginAttributes.emplace_back(PluginField("nbWeights", nullptr, PluginFieldType::kINT32, 1));
                mPluginAttributes.emplace_back(PluginField("eps", nullptr, PluginFieldType::kFLOAT32, 1));
                mFC.nbFields = static_cast<int32_t>(mPluginAttributes.size());
                mFC.fields = mPluginAttributes.data();
                mNamespace = NORMALIZE_PLUGIN_NAME;
            }
            ~NormalizePluginCreator() override = default;
            const char *getPluginName() const noexcept override { return NORMALIZE_PLUGIN_NAME; }
            const char *getPluginVersion() const noexcept override { return NORMALIZE_PLUGIN_VERSION; }
            const PluginFieldCollection *getFieldNames() noexcept override { return &mFC; }
            IPluginV2Ext *createPlugin(const char *name, const PluginFieldCollection *fc) noexcept override {
                std::vector<float> weightValues;
                const PluginField *fields = fc->fields;
                bool mAcrossSpatial{false};
                bool mChannelShared{false};
                float mEps{0.0f};
                int mNbWeights{0};
                for (int i = 0; i < fc->nbFields; ++i) {
                    const char *attrName = fields[i].name;
                    if (!strcmp(attrName, "nbWeights")) {
                        assert(fields[i].type == PluginFieldType::kINT32);
                        mNbWeights = *(static_cast<const int *>(fields[i].data));
                    } else if (!strcmp(attrName, "acrossSpatial")) {
                        assert(fields[i].type == PluginFieldType::kINT32);
                        mAcrossSpatial = *(static_cast<const bool *>(fields[i].data));
                    } else if (!strcmp(attrName, "channelShared")) {
                        assert(fields[i].type == PluginFieldType::kINT32);
                        mChannelShared = *(static_cast<const bool *>(fields[i].data));
                    } else if (!strcmp(attrName, "eps")) {
                        assert(fields[i].type == PluginFieldType::kFLOAT32);
                        mEps = *(static_cast<const float *>(fields[i].data));
                    } else if (!strcmp(attrName, "weights")) {
                        assert(fields[i].type == PluginFieldType::kFLOAT32);
                        int size = fields[i].length;
                        weightValues.reserve(size);
                        const auto *w = static_cast<const float *>(fields[i].data);
                        for (int j = 0; j < size; j++) {
                            weightValues.push_back(*w);
                            w++;
                        }
                    }
                }
                Weights weights{DataType::kFLOAT, weightValues.data(), (int64_t) weightValues.size()};
                auto *plugin = new Normalize(&weights, mNbWeights, mAcrossSpatial, mChannelShared, mEps);
                plugin->setPluginNamespace(mNamespace.c_str());
                return plugin;
            }
            IPluginV2Ext *deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept override {
                auto *plugin = new Normalize(serialData, serialLength);
                plugin->setPluginNamespace(mNamespace.c_str());
                return plugin;
            }

        private:
            PluginFieldCollection mFC{};
            std::vector<PluginField> mPluginAttributes{};
        };
    } // namespace plugin
} // namespace nvinfer1

#endif // TRT_NORMALIZE_PLUGIN_H
