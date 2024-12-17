/*** 
 * @Author: xingwg
 * @Date: 2024-10-12 09:59:37
 * @LastEditTime: 2024-10-12 12:06:33
 * @FilePath: /dmnn2/src/plugin/upsamplePlugin/upsamplePlugin.h
 * @Description: 
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved. 
 */

#ifndef ALGORITHMS_UPSAMPLEPLUGIN_H
#define ALGORITHMS_UPSAMPLEPLUGIN_H

#include <string>
#include <vector>
#include "../common/kernels/kernel.h"
#include "../common/plugin.h"
#include "error_check.h"

using namespace nvinfer1::plugin;

namespace nvinfer1 {
    namespace plugin {
        static const char* Upsample_PLUGIN_VERSION{"1"};
        static const char* Upsample_PLUGIN_NAME{"Upsample_TRT"};
        class Upsample : public IPluginV2Ext {
        public:
            Upsample(int scale) : mScale(scale) { assert(mScale > 0); };
            Upsample(const void *data, size_t length) {
                const char*d = reinterpret_cast<const char*>(data);
                mScale = read<int>(d);
                mInputDims = Dims3();
                mInputDims.d[0] = read<int>(d);
                mInputDims.d[1] = read<int>(d);
                mInputDims.d[2] = read<int>(d);
                mOutputDims = Dims3();
                mOutputDims.d[0] = read<int>(d);
                mOutputDims.d[1] = read<int>(d);
                mOutputDims.d[2] = read<int>(d);
                assert(d == data + length);
            }
            ~Upsample() override = default;
            int getNbOutputs() const noexcept override { return 1; }
            Dims getOutputDimensions(int index, const Dims *inputs, int nbInputDims) noexcept override {
                assert(nbInputDims == 1);
                assert(index == 0);
                nvinfer1::Dims const& input = inputs[0];
                nvinfer1::Dims output;
                output.nbDims = input.nbDims;
                for(int d = 0; d < input.nbDims; ++d) {
                    if(d == input.nbDims -1 || d == input.nbDims - 2) {
                        output.d[d] = input.d[d] * mScale;
                    } else {
                        output.d[d] = input.d[d];
                    }
                }
                return output;
            }
            int initialize() noexcept override { return STATUS_SUCCESS; }
            void terminate() noexcept override {}
            void destroy() noexcept override { delete this; }
            size_t getWorkspaceSize(int maxBatchSize) const noexcept override { return 0; }
            int enqueue(int batchSize, void const* const *inputs, void* const* outputs, void *workspace,
                    cudaStream_t stream) noexcept override;
            size_t getSerializationSize() const noexcept override { return sizeof(int) + sizeof(int) * 3 * 2; }
            void serialize(void *buffer) const noexcept override {
                char *d = reinterpret_cast<char*>(buffer);
                write(d,mScale);
                write(d,mInputDims.d[0]);
                write(d,mInputDims.d[1]);
                write(d,mInputDims.d[2]);
                write(d,mOutputDims.d[0]);
                write(d,mOutputDims.d[1]);
                write(d,mOutputDims.d[2]);
                assert(d == buffer + getSerializationSize());
            }
            bool supportsFormat(DataType type, PluginFormat format) const noexcept override {
                return(type == DataType::kFLOAT && format == PluginFormat::kLINEAR);
            }
            const char *getPluginType() const noexcept override { return Upsample_PLUGIN_NAME; }
            const char *getPluginVersion() const noexcept override { return Upsample_PLUGIN_VERSION; }
            IPluginV2Ext *clone() const noexcept override {
                auto *plugin = new Upsample(mScale);
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
                assert(nbInputs == 1);
                assert(nbOutputs == 1);
                mInputDims = inputDims[0];
                mOutputDims = outputDims[0];
            }
            void detachFromContext() noexcept override {}

        private:
            int mScale;
            Dims mInputDims;
            Dims mOutputDims;
            const char *mPluginNamespace;
        };

        class UpsamplePluginCreator : public BaseCreator {
        public:
            UpsamplePluginCreator() {
                mPluginAttributes.clear();
                mPluginAttributes.emplace_back(PluginField("scale", nullptr,PluginFieldType::kINT32,1));
                mFC.nbFields = mPluginAttributes.size();
                mFC.fields = mPluginAttributes.data();
            }
            ~UpsamplePluginCreator() override = default;
            const char *getPluginName() const noexcept override { return Upsample_PLUGIN_NAME; }
            const char *getPluginVersion() const noexcept override { return Upsample_PLUGIN_VERSION; }
            const PluginFieldCollection *getFieldNames() noexcept override { return &mFC; }
            IPluginV2Ext *createPlugin(const char *name, const PluginFieldCollection *fc) noexcept override {
                const PluginField* fields = fc->fields;
                int scale{0};
                for(int i = 0; i<fc->nbFields ; ++i) {
                    const char* attrName = fields[i].name;
                    if(!strcmp(attrName, "scale")) {
                        assert(fields[i].type==PluginFieldType::kINT32);
                        scale = *(static_cast<const int*>(fields[i].data));
                    }
                }
                auto *plugin = new Upsample(scale);
                plugin->setPluginNamespace(mNamespace.c_str());
                return plugin;
            }

            IPluginV2Ext *deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept override {
                auto *plugin = new Upsample(serialData,serialLength);
                plugin->setPluginNamespace(mNamespace.c_str());
                return plugin;
            }

        private:
            PluginFieldCollection mFC;
            std::vector<PluginField> mPluginAttributes;
        };
    }
}

#endif //ALGORITHMS_UPSAMPLEPLUGIN_H
