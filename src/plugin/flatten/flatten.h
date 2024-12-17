/*** 
 * @Author: xingwg
 * @Date: 2024-10-12 09:59:37
 * @LastEditTime: 2024-10-12 11:29:28
 * @FilePath: /dmnn2/src/plugin/flatten/flatten.h
 * @Description: 
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved. 
 */
#ifndef ALGORITHMS_FLATTEN_H
#define ALGORITHMS_FLATTEN_H
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <NvInferPlugin.h>
#include <cublas_v2.h>
#include <cudnn.h>

#include "../common/plugin.h"
#include "error_check.h"

namespace nvinfer1 {
    namespace plugin {
        static const char* FLATTEN_PLUGIN_VERSION{"1"};
        static const char* FLATTEN_PLUGIN_NAME{"Flatten_TRT"};
        class Flatten : public IPluginV2Ext {
        public:
            Flatten(int channel_in, int height_in, int width_in, int axis, int end_axis)
                    : channel_in_{channel_in}
                    , height_in_{height_in}
                    , width_in_{width_in}
                    , axis_{axis}
                    , end_axis_{end_axis} {}
            Flatten(const void *data, size_t length) {
                const char *d = reinterpret_cast<const char*>(data), *a = d;
                channel_in_ = read<int>(d);
                height_in_ = read<int>(d);
                width_in_ = read<int>(d);
                axis_ = read<int>(d);
                end_axis_ = read<int>(d);
                assert(d == a + length);
            }
            ~Flatten() override = default;
            int getNbOutputs() const noexcept override { return 1; }
            Dims getOutputDimensions(int index, const Dims *inputs, int nbInputDims) noexcept override {
                assert(nbInputDims == 1);
                assert(index == 0);
                int C = inputs[0].d[0]*inputs[0].d[1]*inputs[0].d[2];
                int H = 1;
                int W = 1;
                return Dims3(C, H, W);
            }
            int initialize() noexcept override { return STATUS_SUCCESS; }
            void terminate() noexcept override {}
            size_t getWorkspaceSize(int) const noexcept override { return 0; }
            int enqueue(int batchSize, void const *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept override {
                auto* output = reinterpret_cast<float*>(outputs[0]);
                const auto* input = reinterpret_cast<const float*>(inputs[0]);
                CUDACHECK(cudaMemcpyAsync(output, input, channel_in_*height_in_*width_in_*batchSize*sizeof(float), cudaMemcpyDeviceToDevice, stream));
                return 0;
            }
            DataType getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const noexcept override { return DataType::kFLOAT; }
            size_t getSerializationSize() const noexcept override { return sizeof(int)*5; }
            void serialize(void *buffer) const noexcept override {
                char *d = reinterpret_cast<char*>(buffer), *a = d;
                write(d, channel_in_);
                write(d, height_in_);
                write(d, width_in_);
                write(d, axis_);
                write(d, end_axis_);
                assert(d == a + getSerializationSize());
            }
            bool isOutputBroadcastAcrossBatch(int outputIndex, const bool *inputIsBroadcasted, int nbInputs) const noexcept override { return false; }
            bool canBroadcastInputAcrossBatch(int inputIndex) const noexcept override { return false; }
            void configurePlugin(const Dims *inputDims, int nbInputs, const Dims *outputDims, int nbOutputs,
                                 const DataType *inputTypes, const DataType *outputTypes, const bool *inputIsBroadcast,
                                 const bool *outputIsBroadcast, PluginFormat floatFormat,
                                 int maxBatchSize) noexcept override {
                assert(nbOutputs == 1);
                assert(nbInputs == 1);
                channel_in_ = inputDims[0].d[0];
                height_in_ = inputDims[0].d[1];
                width_in_ = inputDims[0].d[2];
                assert(inputDims[0].nbDims == 3);
            }
            bool supportsFormat(DataType type, PluginFormat format) const noexcept override { return (type == DataType::kFLOAT && format == PluginFormat::kLINEAR); }
            void detachFromContext() noexcept override {}
            const char *getPluginType() const noexcept override { return FLATTEN_PLUGIN_NAME; }
            const char *getPluginVersion() const noexcept override { return FLATTEN_PLUGIN_VERSION; }
            void destroy() noexcept override { delete this; }
            void attachToContext(cudnnContext *cudnnContext, cublasContext *cublasContext, IGpuAllocator *gpuAllocator) noexcept override {}
            IPluginV2Ext *clone() const noexcept override {
                auto* plugin = new Flatten(channel_in_, height_in_, width_in_, axis_, end_axis_);
                plugin->setPluginNamespace(mPluginNamespace);
                return plugin;
            }
            void setPluginNamespace(const char *pluginNamespace) noexcept override { mPluginNamespace = pluginNamespace; }
            const char *getPluginNamespace() const noexcept override { return mPluginNamespace; }
        private:
            int channel_in_;
            int height_in_;
            int width_in_;
            int axis_;
            int end_axis_;
            const char *mPluginNamespace;
        };

        class FlattenPluginCreator : public BaseCreator {
        public:
            FlattenPluginCreator() {
                mPluginAttributes.emplace_back(PluginField("axis", nullptr, PluginFieldType::kINT32, 1));
                mPluginAttributes.emplace_back(PluginField("end_axis", nullptr, PluginFieldType::kINT32, 1));
                mPluginAttributes.emplace_back(PluginField("channel_in", nullptr, PluginFieldType::kINT32, 1));
                mPluginAttributes.emplace_back(PluginField("height_in", nullptr, PluginFieldType::kINT32, 1));
                mPluginAttributes.emplace_back(PluginField("width_in", nullptr, PluginFieldType::kINT32, 1));
                mFC.nbFields = int(mPluginAttributes.size());
                mFC.fields = mPluginAttributes.data();
            }

            ~FlattenPluginCreator() noexcept override = default;
            const char *getPluginName() const noexcept override { return FLATTEN_PLUGIN_NAME; }
            const char *getPluginVersion() const noexcept override { return FLATTEN_PLUGIN_VERSION; }
            const PluginFieldCollection *getFieldNames() noexcept override { return &mFC; }
            IPluginV2Ext *createPlugin(const char *name, const PluginFieldCollection *fc) noexcept override {
                const PluginField* fields = fc->fields;
                int channel_in = 0;
                int height_in = 0;
                int width_in = 0;
                int axis = 0;
                int end_axis = 0;
                for (int i=0; i<fc->nbFields; i++) {
                    const char* attrName = fields[i].name;
                    if (!strcmp(attrName, "axis")) {
                        assert(fields[i].type == PluginFieldType::kINT32);
                        axis = *(static_cast<const int*>(fields[i].data));
                    } else if (!strcmp(attrName, "end_axis")) {
                        assert(fields[i].type == PluginFieldType::kINT32);
                        end_axis = *(static_cast<const int*>(fields[i].data));
                    } else if (!strcmp(attrName, "channel_in")) {
                        assert(fields[i].type == PluginFieldType::kINT32);
                        channel_in = *(static_cast<const int*>(fields[i].data));
                    } else if (!strcmp(attrName, "height_in")) {
                        assert(fields[i].type == PluginFieldType::kINT32);
                        height_in = *(static_cast<const int*>(fields[i].data));
                    } else if (!strcmp(attrName, "width_in")) {
                        assert(fields[i].type == PluginFieldType::kINT32);
                        width_in = *(static_cast<const int*>(fields[i].data));
                    }
                }
                auto* plugin = new Flatten(channel_in, height_in, width_in, axis, end_axis);
                plugin->setPluginNamespace(mNamespace.c_str());
                return plugin;
            }
            IPluginV2Ext *deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept override {
                auto* plugin = new Flatten(serialData, serialLength);
                plugin->setPluginNamespace(mNamespace.c_str());
                return plugin;
            }

        private:
            PluginFieldCollection mFC{};
            std::vector<PluginField> mPluginAttributes{};
        };

    } // namespace plugin
} // namespace nvinfer1

#endif //ALGORITHMS_FLATTEN_H
