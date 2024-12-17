/*** 
 * @Author: xingwg
 * @Date: 2024-10-12 09:59:37
 * @LastEditTime: 2024-10-12 11:37:17
 * @FilePath: /dmnn2/src/plugin/interpPlugin/interpPlugin.h
 * @Description: 
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved. 
 */
#ifndef ALGORITHMS_INTERPPLUGIN_H
#define ALGORITHMS_INTERPPLUGIN_H

#include <vector>
#include <NvInferPlugin.h>
#include <cublas_v2.h>
#include "../common/plugin.h"
#include "error_check.h"

namespace nvinfer1 {
    namespace plugin {
        static const char* INTERP_PLUGIN_VERSION{"1"};
        static const char* INTERP_PLUGIN_NAME{"Interp_TRT"};
        class Interp : public IPluginV2Ext {
        public:
            Interp(int height_out, int width_out, int pad_beg, int pad_end, int height_in, int width_in, int channels)
                    : mOHeight{height_out}, mOWidth{width_out}, mPadBegin{pad_beg}, mPadEnd{pad_end}
                    , mIHeight{width_in}, mIWidth{height_in}, mChannels{channels} {}
            Interp(const void *data, size_t length) {
                const char *d = reinterpret_cast<const char*>(data);
                const char *a = d;
                mOHeight = read<int32_t>(d);
                mOWidth = read<int32_t>(d);
                mIHeight = read<int32_t>(d);
                mIWidth = read<int32_t>(d);
                mPadBegin = read<int32_t>(d);
                mPadEnd = read<int32_t>(d);
                mChannels = read<int32_t>(d);
                assert(d == a + length);
            }
            Interp() = delete;
            ~Interp() override = default;
            int getNbOutputs() const noexcept override { return 1; }
            Dims getOutputDimensions(int index, const Dims *inputs, int nbInputDims) noexcept override {
                assert(nbInputDims == 1);
                assert(index == 0);
                int C = inputs[0].d[0];
                int H = mOHeight;
                int W = mOWidth;
                return Dims3(C, H, W);
            }

            int initialize() noexcept override { return STATUS_SUCCESS; }
            void terminate() noexcept override {}
            size_t getWorkspaceSize(int) const noexcept override { return 0; }
            int enqueue(int32_t batchSize, void const *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept override;
            DataType getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const noexcept override { assert(index == 0); return DataType::kFLOAT; }
            size_t getSerializationSize() const noexcept override { return sizeof(int32_t)*7; }
            void serialize(void *buffer) const noexcept override {
                char *d = reinterpret_cast<char*>(buffer);
                char *a = d;
                write(d, mOHeight);
                write(d, mOWidth);
                write(d, mIHeight);
                write(d, mIWidth);
                write(d, mPadBegin);
                write(d, mPadEnd);
                write(d, mChannels);
                assert(d == a + getSerializationSize());
            }
            bool isOutputBroadcastAcrossBatch(int outputIndex, const bool *inputIsBroadcasted, int nbInputs) const noexcept override { return false; }
            bool canBroadcastInputAcrossBatch(int inputIndex) const noexcept override { return false; }
            void configurePlugin(const Dims *inputDims, int nbInputs, const Dims *outputDims, int nbOutputs,
                                 const DataType *inputTypes, const DataType *outputTypes, const bool *inputIsBroadcast,
                                 const bool *outputIsBroadcast, PluginFormat floatFormat,
                                 int maxBatchSize) noexcept override {
                assert(nbOutputs == 1);
                assert(nbInputs >= 1);
                assert(inputDims[0].nbDims == 3);
                mChannels = inputDims[0].d[0];
                mIHeight = inputDims[0].d[1];
                mIWidth = inputDims[0].d[2];
            }
            bool supportsFormat(DataType type, PluginFormat format) const noexcept override { return (type == DataType::kFLOAT && format == PluginFormat::kLINEAR); }
            void detachFromContext() noexcept override {}
            const char *getPluginType() const noexcept override { return INTERP_PLUGIN_NAME; }
            const char *getPluginVersion() const noexcept override { return INTERP_PLUGIN_VERSION; };
            void destroy() noexcept override { delete this; }
            void attachToContext(cudnnContext *cudnnContext, cublasContext *cublasContext, IGpuAllocator *gpuAllocator) noexcept override {}
            IPluginV2Ext *clone() const noexcept override {
                auto* plugin = new Interp(mOHeight, mOWidth, mPadBegin, mPadEnd, mIHeight, mIWidth, mChannels);
                plugin->setPluginNamespace(mPluginNamespace);
                return plugin;
            }
            void setPluginNamespace(const char *pluginNamespace) noexcept override { mPluginNamespace = pluginNamespace; }
            const char *getPluginNamespace() const noexcept override { return mPluginNamespace; }
        private:
            int mOHeight{0}, mOWidth{0};
            int mIHeight{0}, mIWidth{0};
            int mPadBegin{0}, mPadEnd{0};
            int mChannels{0};
            const char *mPluginNamespace{""};
        };

        class InterpPluginCreator : public BaseCreator {
        public:
            InterpPluginCreator() {
                mPluginAttributes.clear();
                mPluginAttributes.emplace_back(PluginField("channels", nullptr, PluginFieldType::kINT32, 1));
                mPluginAttributes.emplace_back(PluginField("height_in", nullptr, PluginFieldType::kINT32, 1));
                mPluginAttributes.emplace_back(PluginField("width_in", nullptr, PluginFieldType::kINT32, 1));
                mPluginAttributes.emplace_back(PluginField("height_out", nullptr, PluginFieldType::kINT32, 1));
                mPluginAttributes.emplace_back(PluginField("width_out", nullptr, PluginFieldType::kINT32, 1));
                mPluginAttributes.emplace_back(PluginField("pad_beg", nullptr, PluginFieldType::kINT32, 1));
                mPluginAttributes.emplace_back(PluginField("pad_end", nullptr, PluginFieldType::kINT32, 1));
                mFC.nbFields = int(mPluginAttributes.size());
                mFC.fields = mPluginAttributes.data();
            }
            ~InterpPluginCreator() noexcept override = default;
            const char *getPluginName() const noexcept override { return INTERP_PLUGIN_NAME; }
            const char *getPluginVersion() const noexcept override { return INTERP_PLUGIN_VERSION; }
            const PluginFieldCollection *getFieldNames() noexcept override { return &mFC; }
            IPluginV2Ext *createPlugin(const char *name, const PluginFieldCollection *fc) noexcept override {
                const PluginField* fields = fc->fields;
                int channels = 0;
                int height_in = 0;
                int width_in = 0;
                int height_out = 0;
                int width_out = 0;
                int pad_beg = 0;
                int pad_end = 0;
                for (int i=0; i<fc->nbFields; i++) {
                    const char* attrName = fields[i].name;
                    if (!strcmp(attrName, "channels")) {
                        assert(fields[i].type == PluginFieldType::kINT32);
                        channels = *(static_cast<const int*>(fields[i].data));
                    } else if (!strcmp(attrName, "height_in")) {
                        assert(fields[i].type == PluginFieldType::kINT32);
                        height_in = *(static_cast<const int*>(fields[i].data));
                    } else if (!strcmp(attrName, "width_in")) {
                        assert(fields[i].type == PluginFieldType::kINT32);
                        width_in = *(static_cast<const int*>(fields[i].data));
                    } else if (!strcmp(attrName, "height_out")) {
                        assert(fields[i].type == PluginFieldType::kINT32);
                        height_out = *(static_cast<const int*>(fields[i].data));
                    } else if (!strcmp(attrName, "width_out")) {
                        assert(fields[i].type == PluginFieldType::kINT32);
                        width_out = *(static_cast<const int*>(fields[i].data));
                    } else if (!strcmp(attrName, "pad_beg")) {
                        assert(fields[i].type == PluginFieldType::kINT32);
                        pad_beg = *(static_cast<const int*>(fields[i].data));
                    } else if (!strcmp(attrName, "pad_end")) {
                        assert(fields[i].type == PluginFieldType::kINT32);
                        pad_end = *(static_cast<const int*>(fields[i].data));
                    }
                }
                auto* plugin = new Interp(height_out, width_out, pad_beg, pad_end, height_in, width_in, channels);
                plugin->setPluginNamespace(mNamespace.c_str());
                return plugin;
            }
            IPluginV2Ext *deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept override {
                auto* plugin = new Interp(serialData, serialLength);
                plugin->setPluginNamespace(mNamespace.c_str());
                return plugin;
            }

        private:
            PluginFieldCollection mFC{};
            std::vector<PluginField> mPluginAttributes{};
        };
    }
}

#endif //ALGORITHMS_INTERPPLUGIN_H
