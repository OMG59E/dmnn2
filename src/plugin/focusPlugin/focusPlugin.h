/*** 
 * @Author: xingwg
 * @Date: 2024-10-12 09:59:37
 * @LastEditTime: 2024-10-12 12:04:01
 * @FilePath: /dmnn2/src/plugin/focusPlugin/focusPlugin.h
 * @Description: 
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved. 
 */
#ifndef ALGORITHMS_FOCUSPLUGIN_H
#define ALGORITHMS_FOCUSPLUGIN_H

#include <NvInferPlugin.h>
#include <cublas_v2.h>
#include <vector>
#include "../common/plugin.h"
#include "base_types.h"
#include "error_check.h"

namespace nvinfer1 {
    namespace plugin {
        static const char *FOCUS_PLUGIN_VERSION{ "1" };
        static const char *FOCUS_PLUGIN_NAME{ "Focus_TRT" };
        class Focus : public IPluginV2IOExt {
        public:
            explicit Focus(DataType dataType): data_type_{dataType} {};
            Focus(DataType dataType, nv::DimsCHW dims): data_type_{dataType}, input_shape_{dims} {};
            Focus(const void *data, size_t length) {
                const char *d = reinterpret_cast<const char*>(data);
                data_type_ = read<DataType>(d);
                input_shape_ = read<nv::DimsCHW>(d);
                assert(d == data + length);
            }
            ~Focus() override = default;
            int getNbOutputs() const noexcept override { return 1; }
            Dims getOutputDimensions(int index, const Dims *inputs, int nbInputDims) noexcept override {
                assert(nbInputDims == 1);
                assert(index == 0 && inputs[0].nbDims == 3);
                return Dims3{ inputs->d[0]*4, inputs->d[1]/2, inputs->d[2]/2 };
            }
            int initialize() noexcept override { return STATUS_SUCCESS; }
            void terminate() noexcept override { }
            size_t getWorkspaceSize(int maxBatchSize) const noexcept override { return 0; }
            int enqueue(int32_t batchSize, void const* const* inputs, void* const* outputs, void *workspace, cudaStream_t stream) noexcept override;
            DataType getOutputDataType(int index, const DataType *inputTypes, int nbInputs) const noexcept override {
                assert(nbInputs == 1);
                assert(index == 0);
                return inputTypes[0];
            }
            size_t getSerializationSize() const noexcept override { return sizeof(DataType) + sizeof(nv::DimsCHW); }
            void serialize(void *buffer) const noexcept override {
                char *d = reinterpret_cast<char *>(buffer);
                write(d, data_type_);
                write(d, input_shape_);
                assert(d == buffer + getSerializationSize());
            }
            bool isOutputBroadcastAcrossBatch(int outputIndex, const bool *inputIsBroadcasted, int nbInputs) const noexcept override { return false; }
            bool canBroadcastInputAcrossBatch(int inputIndex) const noexcept override { return false; }
            void configurePlugin(const PluginTensorDesc *in, int32_t nbInput, const PluginTensorDesc *out, int32_t nbOutput) noexcept override {
                assert(nbInput == 1 && nbOutput == 1);
                assert(in[0].dims.nbDims == 3);
                assert(in[0].type == out[0].type);
                assert(in[0].format == TensorFormat::kLINEAR && out[0].format == TensorFormat::kLINEAR);
                input_shape_.nbDims = 3;
                input_shape_.d[0] = in[0].dims.d[0];
                input_shape_.d[1] = in[0].dims.d[1];
                input_shape_.d[2] = in[0].dims.d[2];
                data_type_ = in[0].type;
            }

            bool supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut, int32_t nbInputs, int32_t nbOutputs) const noexcept override {
                assert(nbInputs == 1 && nbOutputs == 1 && pos < nbInputs + nbOutputs);
                return inOut[pos].type == inOut[0].type && inOut[pos].format == TensorFormat::kLINEAR;
            }
            void attachToContext(cudnnContext *cudnnContext, cublasContext *cublasContext, IGpuAllocator *gpuAllocator) noexcept override { }
            void detachFromContext() noexcept override {}
            const char *getPluginType() const noexcept override { return FOCUS_PLUGIN_NAME; }
            const char *getPluginVersion() const noexcept override { return FOCUS_PLUGIN_VERSION; }
            void destroy() noexcept override { delete this; }
            IPluginV2IOExt *clone() const noexcept override {
                auto *plugin = new Focus(data_type_, input_shape_);
                plugin->setPluginNamespace(mPluginNamespace.c_str());
                return plugin;
            }
            void setPluginNamespace(const char *pluginNamespace) noexcept override { mPluginNamespace = pluginNamespace; }
            const char *getPluginNamespace() const noexcept override { return mPluginNamespace.c_str(); }
        private:
            nv::DimsCHW input_shape_;
            DataType data_type_{DataType::kFLOAT};
            std::string mPluginNamespace;
        };

        class FocusPluginCreator : public BaseCreator {
        public:
            FocusPluginCreator() { mNamespace = FOCUS_PLUGIN_NAME; }
            ~FocusPluginCreator() override = default;
            const char *getPluginName() const noexcept override { return FOCUS_PLUGIN_NAME; }
            const char *getPluginVersion() const noexcept override { return FOCUS_PLUGIN_VERSION; }
            const PluginFieldCollection* getFieldNames() noexcept override { return &mFC; }
            IPluginV2IOExt *createPlugin(const char *name, const PluginFieldCollection *fc) noexcept override {
                auto *plugin = new Focus(DataType::kFLOAT);
                plugin->setPluginNamespace(mNamespace.c_str());
                return plugin;
            }
            IPluginV2IOExt *deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept override {
                auto *plugin = new Focus(serialData, serialLength);
                plugin->setPluginNamespace(mNamespace.c_str());
                return plugin;
            }
        private:
            PluginFieldCollection mFC{};
            std::vector<PluginField> mPluginAttributes{};
        };
    }
}

#endif //ALGORITHMS_FOCUSPLUGIN_H
