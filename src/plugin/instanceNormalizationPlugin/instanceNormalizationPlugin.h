/*** 
 * @Author: xingwg
 * @Date: 2024-10-12 09:59:37
 * @LastEditTime: 2024-10-12 11:31:28
 * @FilePath: /dmnn2/src/plugin/instanceNormalizationPlugin/instanceNormalizationPlugin.h
 * @Description: 
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved. 
 */
#ifndef TRT_INSTANCE_NORMALIZATION_PLUGIN_H
#define TRT_INSTANCE_NORMALIZATION_PLUGIN_H
#include <cuda_fp16.h>
#include <cudnn.h>
#include <iostream>
#include <string>
#include <vector>

#include "instanceNormFwd.h"
#include "../common/plugin.h"
#include "../common/serialize.hpp"
#include "error_check.h"

typedef unsigned short half_type;

namespace nvinfer1 {
    namespace plugin {
        static const char *INSTANCE_PLUGIN_VERSION{"1"};
        static const char *INSTANCE_PLUGIN_NAME{"InstanceNormalization_TRT"};
        using namespace instance_norm_impl;
        class InstanceNormalizationPlugin final : public nvinfer1::IPluginV2DynamicExt {
        public:
            InstanceNormalizationPlugin(float epsilon, nvinfer1::Weights const &scale, nvinfer1::Weights const &bias, int32_t relu = 0, float alpha = 0.f)
                    : mEpsilon(epsilon), mAlpha(alpha), mRelu(relu), mNchan(scale.count) {
                assert(scale.count == bias.count);
                const auto copyWeights = [](nvinfer1::Weights const &input, std::vector<float> &output) {
                    output.reserve(input.count);
                    if (input.type == nvinfer1::DataType::kFLOAT) {
                        output.assign(static_cast<const float *>(input.values), static_cast<const float *>(input.values) + input.count);
                    } else if (input.type == nvinfer1::DataType::kHALF) {
                        for (int32_t c = 0; c < input.count; ++c) {
                            const auto value = static_cast<const unsigned short*>(input.values);
                            output.push_back(__internal_half2float(value[c]));
                        }
                    } else {
                        throw std::runtime_error("Unsupported scale/bias dtype");
                    }
                };
                copyWeights(scale, mHostScale);
                copyWeights(bias, mHostBias);
            }
            InstanceNormalizationPlugin(float epsilon, const std::vector<float> &scale, const std::vector<float> &bias, int32_t relu = 0, float alpha = 0.f)
                    : mEpsilon(epsilon), mAlpha(alpha), mRelu(relu), mNchan(scale.size()), mHostScale(scale), mHostBias(bias) {
                assert(scale.size() == bias.size());
            }
            InstanceNormalizationPlugin(const void* serialData, size_t serialLength) {
                deserialize_value(&serialData, &serialLength, &mEpsilon);
                deserialize_value(&serialData, &serialLength, &mNchan);
                deserialize_value(&serialData, &serialLength, &mHostScale);
                deserialize_value(&serialData, &serialLength, &mHostBias);
                deserialize_value(&serialData, &serialLength, &mRelu);
                deserialize_value(&serialData, &serialLength, &mAlpha);
            }
            InstanceNormalizationPlugin() = delete;
            ~InstanceNormalizationPlugin() override { terminate(); }
            int32_t getNbOutputs() const noexcept override { return 1; }
            DimsExprs getOutputDimensions(int32_t outputIndex, const nvinfer1::DimsExprs *inputs, int32_t nbInputs, nvinfer1::IExprBuilder &exprBuilder) noexcept override {
                nvinfer1::DimsExprs output(inputs[0]);
                return output;
            }
            int32_t initialize() noexcept override {
                if (!mInitialized) {
                    CUDNNCHECK(cudnnCreate(&mCudnnHandle));
                    CUDNNCHECK(cudnnCreateTensorDescriptor(&mBDescriptor));
                    CUDNNCHECK(cudnnCreateTensorDescriptor(&mXDescriptor));
                    CUDNNCHECK(cudnnCreateTensorDescriptor(&mYDescriptor));
                    // NDHWC path
                    // Device info.
                    int32_t device;
                    CUDACHECK(cudaGetDevice(&device));
                    cudaDeviceProp props;
                    CUDACHECK(cudaGetDeviceProperties(&props, device));
                    mContext.sm_count = props.multiProcessorCount;
                    mContext.sm_shared_size = props.sharedMemPerMultiprocessor;
                    mContext.sm_version = props.major * 100 + props.minor * 10;
                    CUDACHECK(cudaMalloc(&mDeviceScale, mNchan * sizeof(float)));
                    CUDACHECK(cudaMalloc(&mDeviceBias, mNchan * sizeof(float)));
                    CUDACHECK(cudaMemcpy(mDeviceScale, &mHostScale[0], mNchan * sizeof(float), cudaMemcpyHostToDevice));
                    CUDACHECK(cudaMemcpy(mDeviceBias, &mHostBias[0], mNchan * sizeof(float), cudaMemcpyHostToDevice));
                }
                mInitialized = true;
                return 0;
            }
            void terminate() noexcept override {
                if (mInitialized) {
                    cudnnDestroyTensorDescriptor(mYDescriptor);
                    cudnnDestroyTensorDescriptor(mXDescriptor);
                    cudnnDestroyTensorDescriptor(mBDescriptor);
                    cudnnDestroy(mCudnnHandle);
                    CUDACHECK(cudaFree(mDeviceBias));
                    CUDACHECK(cudaFree(mDeviceScale));
                }
                mInitialized = false;
            }
            size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs, int32_t nbInputs, const nvinfer1::PluginTensorDesc *outputs, int32_t nbOutputs) const noexcept override {
                nvinfer1::Dims input_dims = inputs[0].dims;
                assert(input_dims.nbDims == 4 || input_dims.nbDims == 5);
                if (inputs[0].format == nvinfer1::PluginFormat::kLINEAR) {
                    nvinfer1::Dims input_dims = inputs[0].dims;
                    int32_t n = input_dims.d[0];
                    int32_t c = input_dims.d[1];
                    size_t nchan_bytes = c * sizeof(float);
                    size_t scale_size = n * nchan_bytes;
                    size_t bias_size = n * nchan_bytes;
                    size_t total_wss = scale_size + bias_size;
                    return total_wss;
                } else if (inputs[0].format == nvinfer1::PluginFormat::kDHWC8 || inputs[0].format == nvinfer1::PluginFormat::kCDHW32) {
                    assert(input_dims.nbDims == 5);
                    int32_t input_data_type = (inputs[0].type == nvinfer1::DataType::kHALF) ? 1 : 2;
                    int32_t output_data_type = (outputs[0].type == nvinfer1::DataType::kHALF) ? 1 : 2;
                    nvinfer1::Dims input_dims = inputs[0].dims;

                    int32_t n = input_dims.d[0];
                    int32_t c = input_dims.d[1];
                    int32_t d = input_dims.d[2];
                    int32_t h = input_dims.d[3];
                    int32_t w = input_dims.d[4];

                    InstanceNormFwdParams params;
                    // only these parameters are required for workspace computation
                    params.nhw = d * h * w;
                    params.c = c;
                    params.n = n;
                    // Reserve memory for the workspaces.
                    size_t size_sums, size_counts, size_retired_ctas;
                    instanceNormBufferSizesDispatch(mContext, params, size_sums, size_counts, size_retired_ctas, input_data_type, output_data_type);
                    size_t size_nc = n * c * sizeof(float);
                    size_nc = ((size_nc + 256 - 1) / 256) * 256;
                    return size_sums + size_counts + size_retired_ctas + 4 * size_nc;
                } else {
                    assert(0);
                }
                return 0;
            }
            int32_t enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
                            const void *const *inputs, void *const *outputs, void *workspace,
                            cudaStream_t stream) noexcept override;
            size_t getSerializationSize() const noexcept override {
                return (serialized_size(mEpsilon) + serialized_size(mNchan) + serialized_size(mHostScale)
                        + serialized_size(mHostBias) + serialized_size(mRelu) + serialized_size(mAlpha));
            }
            void serialize(void *buffer) const noexcept override {
                serialize_value(&buffer, mEpsilon);
                serialize_value(&buffer, mNchan);
                serialize_value(&buffer, mHostScale);
                serialize_value(&buffer, mHostBias);
                serialize_value(&buffer, mRelu);
                serialize_value(&buffer, mAlpha);
            }

            // DynamicExt plugin supportsFormat update.
            bool supportsFormatCombination(int32_t pos, const nvinfer1::PluginTensorDesc *inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override {
                assert(inOut && pos < (nbInputs + nbOutputs));
                assert(pos == 0 || pos == 1);

                // For 4-D or 3-D tensor (nbSpatialDims == 1 or 2), only FP32_Linear and FP16_Linear are supported.
                // For 5-D tensor (nbSpatialDims == 3), FP32_Linear, FP16_Linear, FP16_DHWC8, and INT8_CDHW32 are supported.
                // This is because we have special InstanceNorm3D kernels for vectorized formats from MLPerf-Inference.

                const int32_t nbDims = inOut[pos].dims.nbDims;
                assert(nbDims >= 3);
                assert(nbDims <= 5);
                const bool is3DInstanceNorm = (nbDims == 5);

                const bool isFP32Linear = (inOut[pos].type == nvinfer1::DataType::kFLOAT && inOut[pos].format == nvinfer1::PluginFormat::kLINEAR
                           && inOut[pos].type == inOut[0].type && inOut[pos].format == inOut[0].format);
                const bool isFP16Linear = (inOut[pos].type == nvinfer1::DataType::kHALF && inOut[pos].format == nvinfer1::PluginFormat::kLINEAR
                           && inOut[pos].type == inOut[0].type && inOut[pos].format == inOut[0].format);
                const bool isFP16DHWC8 = (inOut[pos].type == nvinfer1::DataType::kHALF && inOut[pos].format == nvinfer1::PluginFormat::kDHWC8
                           && inOut[pos].type == inOut[0].type && inOut[pos].format == inOut[0].format);
                const bool isINT8CDHW32 = (inOut[pos].type == nvinfer1::DataType::kINT8 && inOut[pos].format == nvinfer1::PluginFormat::kCDHW32
                           && inOut[pos].type == inOut[0].type && inOut[pos].format == inOut[0].format);
                const bool isFormatOK = isFP32Linear || isFP16Linear || (is3DInstanceNorm && (isFP16DHWC8 || isINT8CDHW32));

                // Kernels for vectorized formats only support the case of C % spv == 0.
                int32_t spv{1};
                switch (inOut[pos].format) {
                    case nvinfer1::PluginFormat::kDHWC8:
                        spv = 8;
                        break;
                    case nvinfer1::PluginFormat::kCDHW32:
                        spv = 32;
                        break;
                    default:
                        break;
                }
                const int32_t isAlignmentOK = (inOut[pos].dims.d[1] % spv == 0);
                return isFormatOK && isAlignmentOK;
            }

            const char *getPluginType() const noexcept override { return INSTANCE_PLUGIN_NAME; }
            const char *getPluginVersion() const noexcept override { return INSTANCE_PLUGIN_VERSION; }
            void destroy() noexcept override { delete this; }
            nvinfer1::IPluginV2DynamicExt *clone() const noexcept override {
                auto *plugin = new InstanceNormalizationPlugin{mEpsilon, mHostScale, mHostBias, mRelu, mAlpha};
                plugin->setPluginNamespace(mPluginNamespace.c_str());
                plugin->initialize();
                return plugin;
            }
            void setPluginNamespace(const char *pluginNamespace) noexcept override { mPluginNamespace = pluginNamespace; }
            const char *getPluginNamespace() const noexcept override { return mPluginNamespace.c_str(); }
            DataType getOutputDataType(int32_t index, const nvinfer1::DataType *inputTypes, int32_t nbInputs) const noexcept override {
                assert(inputTypes && nbInputs > 0 && index == 0);
                return inputTypes[0];
            }
            void attachToContext(cudnnContext *cudnn, cublasContext *cublas, nvinfer1::IGpuAllocator *allocator) noexcept override {}
            void detachFromContext() noexcept override {}
            void configurePlugin(const nvinfer1::DynamicPluginTensorDesc *in, int32_t nbInputs,
                                 const nvinfer1::DynamicPluginTensorDesc *out, int32_t nbOutputs) noexcept override {
                // Not support dynamic shape in C dimension
                assert(nbInputs == 1 && in[0].desc.dims.d[1] != -1);
            }

        private:
            float mEpsilon;
            float mAlpha;
            int32_t mRelu;
            int32_t mNchan;
            std::vector<float> mHostScale;
            std::vector<float> mHostBias;
            float *mDeviceScale{nullptr};
            float *mDeviceBias{nullptr};
            cudnnHandle_t mCudnnHandle{nullptr};
            cudnnTensorDescriptor_t mXDescriptor{nullptr};
            cudnnTensorDescriptor_t mYDescriptor{nullptr};
            cudnnTensorDescriptor_t mBDescriptor{nullptr};
            std::string mPluginNamespace;
            std::string mNamespace;
            bool mInitialized{false};
            // NDHWC implementation
            InstanceNormFwdContext mContext;
        };

        class InstanceNormalizationPluginCreator : public BaseCreator {
        public:
            InstanceNormalizationPluginCreator() {
                mPluginAttributes.clear();
                mPluginAttributes.emplace_back(PluginField("epsilon", nullptr, PluginFieldType::kFLOAT32, 1));
                mPluginAttributes.emplace_back(PluginField("scales", nullptr, PluginFieldType::kFLOAT32, 1));
                mPluginAttributes.emplace_back(PluginField("bias", nullptr, PluginFieldType::kFLOAT32, 1));
                mPluginAttributes.emplace_back(PluginField("relu", nullptr, PluginFieldType::kINT32, 1));
                mPluginAttributes.emplace_back(PluginField("alpha", nullptr, PluginFieldType::kFLOAT32, 1));
                mFC.nbFields = mPluginAttributes.size();
                mFC.fields = mPluginAttributes.data();
            }
            ~InstanceNormalizationPluginCreator() override = default;
            const char *getPluginName() const noexcept override { return INSTANCE_PLUGIN_NAME; }
            const char *getPluginVersion() const noexcept override { return INSTANCE_PLUGIN_VERSION; }
            const PluginFieldCollection *getFieldNames() noexcept override { return &mFC; }
            IPluginV2DynamicExt *createPlugin(const char *name, const nvinfer1::PluginFieldCollection *fc) noexcept override {
                std::vector<float> scaleValues;
                std::vector<float> biasValues;
                float epsilon{};
                int32_t relu{};
                float alpha{};
                const PluginField *fields = fc->fields;
                for (int32_t i = 0; i < fc->nbFields; ++i) {
                    const char *attrName = fields[i].name;
                    if (!strcmp(attrName, "epsilon")) {
                        assert(fields[i].type == PluginFieldType::kFLOAT32);
                        epsilon = *(static_cast<const float *>(fields[i].data));
                    } else if (!strcmp(attrName, "scales")) {
                        assert(fields[i].type == PluginFieldType::kFLOAT32);
                        int32_t size = fields[i].length;
                        scaleValues.reserve(size);
                        const auto *w = static_cast<const float *>(fields[i].data);
                        for (int32_t j = 0; j < size; j++) {
                            scaleValues.push_back(*w);
                            w++;
                        }
                    } else if (!strcmp(attrName, "bias")) {
                        assert(fields[i].type == PluginFieldType::kFLOAT32);
                        int32_t size = fields[i].length;
                        biasValues.reserve(size);
                        const auto *w = static_cast<const float *>(fields[i].data);
                        for (int32_t j = 0; j < size; j++) {
                            biasValues.push_back(*w);
                            w++;
                        }
                    } else if (!strcmp(attrName, "relu")) {
                        assert(fields[i].type == PluginFieldType::kINT32);
                        relu = *(static_cast<const int32_t *>(fields[i].data));
                    } else if (!strcmp(attrName, "alpha")) {
                        assert(fields[i].type == PluginFieldType::kFLOAT32);
                        alpha = *(static_cast<const float *>(fields[i].data));
                    }
                }

                Weights scaleWeights{DataType::kFLOAT, scaleValues.data(), (int64_t) scaleValues.size()};
                Weights biasWeights{DataType::kFLOAT, biasValues.data(), (int64_t) biasValues.size()};

                auto *plugin = new InstanceNormalizationPlugin(epsilon, scaleWeights, biasWeights, relu, alpha);
                plugin->setPluginNamespace(mNamespace.c_str());
                plugin->initialize();
                return plugin;
            }
            IPluginV2DynamicExt *deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept override {
                auto *plugin = new InstanceNormalizationPlugin(serialData, serialLength);
                plugin->setPluginNamespace(mNamespace.c_str());
                plugin->initialize();
                return plugin;
            }
        private:
            PluginFieldCollection mFC{};
            std::vector<PluginField> mPluginAttributes{};
        };
    } // namespace plugin
} // namespace nvinfer1

#endif // TRT_INSTANCE_NORMALIZATION_PLUGIN_H
