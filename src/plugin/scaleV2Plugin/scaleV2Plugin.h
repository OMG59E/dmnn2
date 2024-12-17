/*** 
 * @Author: xingwg
 * @Date: 2024-10-12 09:59:37
 * @LastEditTime: 2024-10-12 14:04:08
 * @FilePath: /dmnn2/src/plugin/scaleV2Plugin/scaleV2Plugin.h
 * @Description: 
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved. 
 */
#ifndef ALGORITHMS_SCALEV2PLUGIN_H
#define ALGORITHMS_SCALEV2PLUGIN_H

#include <vector>
#include <NvInferPlugin.h>
#include <cublas_v2.h>
#include "../common/plugin.h"
#include "error_check.h"

namespace nvinfer1 {
    namespace plugin {
        static const char* SCALEV2_PLUGIN_VERSION{"1"};
        static const char* SCALEV2_PLUGIN_NAME{"ScaleV2_TRT"};
        class ScaleV2 : public IPluginV2IOExt {
        public:
            ScaleV2(int axis, int num_axes, int bias_term)
                : axis_{axis}, num_axes_{num_axes}, bias_term_{bias_term} {}
            ScaleV2(int C, int H, int W, DataType data_type, int axis, int num_axes, int bias_term)
                : axis_{axis}, num_axes_{num_axes}, bias_term_{bias_term}
                , channel_in_{C}, height_in_{H}, width_in_{W}, data_type_{data_type} {}
            ScaleV2(const void* data, size_t length) {
                const char *d = reinterpret_cast<const char*>(data);
                axis_ = read<int32_t>(d);
                num_axes_ = read<int32_t>(d);
                bias_term_ = read<int32_t>(d);
                channel_in_ = read<int32_t>(d);
                height_in_  = read<int32_t>(d);
                width_in_   = read<int32_t>(d);
                data_type_ = read<DataType>(d);
                assert(d == data + length);
            }
            ScaleV2() = delete;
            ~ScaleV2() override = default;
            int getNbOutputs() const noexcept override { return 1; }
            Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept override {
                assert(nbInputDims == 2);
                assert(inputs[0].nbDims == 3);
                assert(inputs[1].nbDims == 3 || inputs[1].nbDims == 1);
                assert(index == 0);
                if (inputs[1].nbDims == 3) {
                    assert(inputs[1].d[1] == 1);
                    assert(inputs[1].d[2] == 1);
                }
                channel_in_ = inputs[0].d[0];
                height_in_  = inputs[0].d[1];
                width_in_   = inputs[0].d[2];
                return Dims3(channel_in_, height_in_, width_in_);
            }
            int initialize() noexcept override { return STATUS_SUCCESS; }
            void terminate() noexcept override {}
            size_t getWorkspaceSize(int) const noexcept override { return 0; }
            int enqueue(int32_t batchSize,  void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
            DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept override {
                assert(nbInputs == 2);
                assert(index == 0);
                return inputTypes[0];
            }
            size_t getSerializationSize() const noexcept override { return sizeof(int32_t)*6 + sizeof(DataType); }
            void serialize(void* buffer) const noexcept override {
                char *d = reinterpret_cast<char*>(buffer);
                write(d, axis_);
                write(d, num_axes_);
                write(d, bias_term_);
                write(d, channel_in_);
                write(d, height_in_);
                write(d, width_in_);
                write(d, data_type_);
                assert(d == buffer + getSerializationSize());
            }
            bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const noexcept override { return false; }
            bool canBroadcastInputAcrossBatch(int inputIndex) const noexcept override { return false; }
            void configurePlugin(PluginTensorDesc const* in, int32_t nbInput, PluginTensorDesc const* out, int32_t nbOutput) noexcept override {
                assert(nbInput == 2 && nbOutput == 1);
                assert(in[0].dims.nbDims == 3);
                assert(in[0].type == out[0].type);
                assert(in[0].format == TensorFormat::kLINEAR && out[0].format == TensorFormat::kLINEAR);
                if (in[1].dims.nbDims == 3) {
                    assert(in[1].dims.d[1] == 1);
                    assert(in[1].dims.d[2] == 1);
                }
                assert(in[0].dims.d[0] == in[1].dims.d[0]);
                channel_in_ = in[0].dims.d[0];
                height_in_ = in[0].dims.d[1];
                width_in_ = in[0].dims.d[2];
                data_type_ = in[0].type;
            }
            bool supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut, int32_t nbInputs, int32_t nbOutputs) const noexcept override {
                assert(nbInputs == 2 && nbOutputs == 1 && pos < nbInputs + nbOutputs);
                return inOut[pos].type == inOut[0].type && inOut[pos].format == TensorFormat::kLINEAR;
            }
            void detachFromContext() noexcept override {}
            const char* getPluginType() const noexcept override { return SCALEV2_PLUGIN_NAME; }
            const char* getPluginVersion() const noexcept override { return SCALEV2_PLUGIN_VERSION; }
            void destroy() noexcept override { delete this; }
            void attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept override {}
            IPluginV2IOExt* clone() const noexcept override {
                auto *plugin = new ScaleV2(channel_in_, height_in_, width_in_, data_type_, axis_, num_axes_, bias_term_);
                plugin->setPluginNamespace(mPluginNamespace);
                return plugin;
            }
            void setPluginNamespace(const char* pluginNamespace) noexcept override { mPluginNamespace = pluginNamespace; }
            const char* getPluginNamespace() const noexcept override { return mPluginNamespace; }

        private:
            DataType data_type_{DataType::kFLOAT};
            int axis_{-1};
            int num_axes_{-1};
            int bias_term_{-1};
            int channel_in_{-1};
            int height_in_{-1};
            int width_in_{-1};
            //int scale_dim_{-1};
            const char* mPluginNamespace;
        };

        class ScaleV2PluginCreator : public BaseCreator {
        public:
            ScaleV2PluginCreator() {
                mPluginAttributes.clear();
                mPluginAttributes.emplace_back(PluginField("axis", nullptr, PluginFieldType::kINT32, 1));
                mPluginAttributes.emplace_back(PluginField("num_axes", nullptr, PluginFieldType::kINT32, 1));
                mPluginAttributes.emplace_back(PluginField("bias_term", nullptr, PluginFieldType::kINT32, 1));
                mFC.nbFields = int(mPluginAttributes.size());
                mFC.fields = mPluginAttributes.data();
            }
            ~ScaleV2PluginCreator() override = default;
            const char* getPluginName() const noexcept override { return SCALEV2_PLUGIN_NAME; }
            const char* getPluginVersion() const noexcept override { return SCALEV2_PLUGIN_VERSION; }
            const PluginFieldCollection* getFieldNames() noexcept override { return &mFC; }
            IPluginV2IOExt* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override {
                const PluginField* fields = fc->fields;
                int axis = 0;
                int num_axes = 0;
                int bias_term = 0;
                for (int i=0; i<fc->nbFields; i++) {
                    const char* attrName = fields[i].name;
                    if (!strcmp(attrName, "axis")) {
                        assert(fields[i].type == PluginFieldType::kINT32);
                        axis = *(static_cast<const int*>(fields[i].data));
                    } else if (!strcmp(attrName, "num_axes")) {
                        assert(fields[i].type == PluginFieldType::kINT32);
                        num_axes = *(static_cast<const int*>(fields[i].data));
                    } else if (!strcmp(attrName, "bias_term")) {
                        assert(fields[i].type == PluginFieldType::kINT32);
                        bias_term = *(static_cast<const int*>(fields[i].data));
                    }
                }
                auto *plugin = new ScaleV2(axis, num_axes, bias_term);
                plugin->setPluginNamespace(mNamespace.c_str());
                return plugin;
            }
            IPluginV2IOExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override {
                auto *plugin = new ScaleV2(serialData, serialLength);
                plugin->setPluginNamespace(mNamespace.c_str());
                return plugin;
            }

        private:
            PluginFieldCollection mFC{};
            std::vector<PluginField> mPluginAttributes{};
        };
    }
}

#endif //ALGORITHMS_SCALEV2PLUGIN_H
