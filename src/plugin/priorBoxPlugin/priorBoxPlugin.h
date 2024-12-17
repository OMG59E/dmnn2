/*** 
 * @Author: xingwg
 * @Date: 2024-10-12 09:59:37
 * @LastEditTime: 2024-10-12 11:39:32
 * @FilePath: /dmnn2/src/plugin/priorBoxPlugin/priorBoxPlugin.h
 * @Description: 
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved. 
 */
#ifndef TRT_PRIOR_BOX_PLUGIN_H
#define TRT_PRIOR_BOX_PLUGIN_H

#include <string>
#include <vector>
#include <cstdlib>
#include <cublas_v2.h>
#include <cudnn.h>
#include "../common/kernels/kernel.h"
#include "../common/plugin.h"
#include "error_check.h"

namespace nvinfer1 {
    namespace plugin {
        static const char* PRIOR_BOX_PLUGIN_VERSION{"1"};
        static const char* PRIOR_BOX_PLUGIN_NAME{"PriorBox_TRT"};
        class PriorBox : public IPluginV2Ext {
        public:
            explicit PriorBox(PriorBoxParameters param)
                : mParam(param), mOwnsParamMemory(true) {
                // minSize is required and needs to be non-negative
                assert(param.numMinSize > 0 && param.minSize != nullptr);
                for (int i = 0; i < param.numMinSize; ++i)
                    assert(param.minSize[i] > 0 && "minSize must be positive");
                minSize = copyToDevice(param.minSize, param.numMinSize);
                assert(param.numAspectRatios >= 0 && param.aspectRatios != nullptr);
                // Aspect ratio of 1.0 is built in.
                std::vector<float> tmpAR(1, 1);
                for (int i = 0; i < param.numAspectRatios; ++i) {
                    float ar = param.aspectRatios[i];
                    bool alreadyExist = false;
                    // Prevent duplicated aspect ratios from input
                    for (unsigned j = 0; j < tmpAR.size(); ++j) {
                        if (std::fabs(ar - tmpAR[j]) < 1e-6) {
                            alreadyExist = true;
                            break;
                        }
                    }
                    if (!alreadyExist) {
                        tmpAR.push_back(ar);
                        if (param.flip)
                            tmpAR.push_back(1.0F / ar);
                    }
                }
                /*
                 * aspectRatios is of type nvinfer1::Weights
                 * https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/c_api/classnvinfer1_1_1_weights.html
                 * aspectRatios.count is different to param.numAspectRatios
                 */
                aspectRatios = copyToDevice(&tmpAR[0], tmpAR.size());
                // Number of prior boxes per grid cell on the feature map
                // tmpAR already included an aspect ratio of 1.0
                numPriors = tmpAR.size() * param.numMinSize;
                /*
                 * If we have maxSizes, as long as all the maxSizes meets assertion requirement, we add one bounding box per maxSize
                 * The final number of prior boxes per grid cell on feature map
                 * numPriors =
                 * tmpAR.size() * param.numMinSize If numMaxSize == 0
                 * (tmpAR.size() + 1) * param.numMinSize If param.numMinSize == param.numMaxSize
                 */
                if (param.numMaxSize > 0) {
                    assert(param.numMinSize == param.numMaxSize && param.maxSize != nullptr);
                    for (int i = 0; i < param.numMaxSize; ++i) {
                        // maxSize should be greater than minSize
                        assert(param.maxSize[i] > param.minSize[i] && "maxSize must be greater than minSize");
                        numPriors++;
                    }
                    maxSize = copyToDevice(param.maxSize, param.numMaxSize);
                }
            }

            PriorBox(PriorBoxParameters param, int numPriors, int H, int W,
                    Weights minSize, Weights maxSize, Weights aspectRatios)
                    : mParam(param), mOwnsParamMemory(false), numPriors(numPriors)
                    , H(H), W(W), minSize(minSize), maxSize(maxSize), aspectRatios(aspectRatios) {}

            PriorBox(const void *buffer, size_t length)
                    : mOwnsParamMemory(true) {
                const char *d = reinterpret_cast<const char*>(buffer), *a = d;
                mParam = read<PriorBoxParameters>(d);
                mParam.minSize = new float[mParam.numMinSize];
                mParam.maxSize = new float[mParam.numMaxSize];

                numPriors = read<int>(d);
                H = read<int>(d);
                W = read<int>(d);

                for (auto i = 0; i < mParam.numMinSize; i++)
                    mParam.minSize[i] = reinterpret_cast<const float*>(d)[i];
                minSize = deserializeToDevice(d, mParam.numMinSize);

                if (mParam.numMaxSize > 0) {
                    for (auto i = 0; i < mParam.numMaxSize; i++)
                        mParam.maxSize[i] = reinterpret_cast<const float*>(d)[i];
                    maxSize = deserializeToDevice(d, mParam.numMaxSize);
                }

                int numAspectRatios = read<int>(d);
                mParam.aspectRatios = new float[numAspectRatios];
                if (numAspectRatios > 0) {
                    for (auto i = 0; i < numAspectRatios; i++)
                        mParam.aspectRatios[i] = reinterpret_cast<const float*>(d)[i];
                    aspectRatios = deserializeToDevice(d, numAspectRatios);
                }
                assert(d == a + length);
            }
            ~PriorBox() override = default;
            int getNbOutputs() const noexcept override { return 1; }
            Dims getOutputDimensions(int index, const Dims *inputs, int nbInputDims) noexcept override {
                assert(nbInputDims == 2);
                // Only one output from the plugin layer
                assert(index == 0);
                // Particularity of the PriorBox layer: no batchSize dimension needed
                H = inputs[0].d[1], W = inputs[0].d[2];
                // workaround for TRT
                // The first channel is for prior box coordinates.
                // The second channel is for prior box scaling factors, which is simply a copy of the variance provided.
                return Dims3(2, H * W * numPriors * 4, 1);
            }
            int initialize() noexcept override { return STATUS_SUCCESS; }
            void terminate() noexcept override {
                if (mOwnsParamMemory) {
                    CUDACHECK(cudaFree(const_cast<void*>(minSize.values)));
                    if (mParam.numMaxSize > 0)
                        CUDACHECK(cudaFree(const_cast<void*>(maxSize.values)));
                    if (mParam.numAspectRatios >= 0)
                        CUDACHECK(cudaFree(const_cast<void*>(aspectRatios.values)));
                    delete[] mParam.minSize;
                    delete[] mParam.maxSize;
                    delete[] mParam.aspectRatios;
                }
            }
            size_t getWorkspaceSize(int maxBatchSize) const noexcept override { return 0; }
            int enqueue(int batchSize, void const* const* inputs, void* const* outputs, void *workspace,
                    cudaStream_t stream) noexcept override {
                void* outputData = outputs[0];
                pluginStatus_t status = priorBoxInference(stream, mParam, H, W, numPriors, aspectRatios.count, minSize.values,
                                                          maxSize.values, aspectRatios.values, outputData);
                assert(status == STATUS_SUCCESS);
                return 0;
            }
            size_t getSerializationSize() const noexcept override {
                return sizeof(PriorBoxParameters) + sizeof(int) * 3 + sizeof(float) * (mParam.numMinSize + mParam.numMaxSize)
                       + sizeof(int) + sizeof(float) * aspectRatios.count;
            }
            void serialize(void *buffer) const noexcept override {
                char *d = reinterpret_cast<char*>(buffer), *a = d;
                write(d, mParam);
                write(d, numPriors);
                write(d, H);
                write(d, W);
                serializeFromDevice(d, minSize);
                if (mParam.numMaxSize > 0)
                    serializeFromDevice(d, maxSize);
                write(d, (int) aspectRatios.count);
                serializeFromDevice(d, aspectRatios);
                assert(d == a + getSerializationSize());
            }
            bool supportsFormat(DataType type, PluginFormat format) const noexcept override {
                return (type == DataType::kFLOAT && format == PluginFormat::kLINEAR);
            }
            const char *getPluginType() const noexcept override { return PRIOR_BOX_PLUGIN_NAME; }
            const char *getPluginVersion() const noexcept override { return PRIOR_BOX_PLUGIN_VERSION; }
            void destroy() noexcept override { delete this; }
            IPluginV2Ext *clone() const noexcept override {
                auto* plugin = new PriorBox(mParam, numPriors, H, W, minSize, maxSize, aspectRatios);
                plugin->setPluginNamespace(mPluginNamespace.c_str());
                return plugin;
            }
            void setPluginNamespace(const char *pluginNamespace) noexcept override { mPluginNamespace = pluginNamespace; }
            const char *getPluginNamespace() const noexcept override { return mPluginNamespace.c_str(); }
            DataType getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const noexcept override {
                assert(index == 0 || index == 1);
                return DataType::kFLOAT;
            }
            bool isOutputBroadcastAcrossBatch(int outputIndex, const bool *inputIsBroadcasted, int nbInputs) const noexcept override { return false; }
            bool canBroadcastInputAcrossBatch(int inputIndex) const noexcept override { return false; }
            void attachToContext(cudnnContext *cudnnContext, cublasContext *cublasContext, IGpuAllocator *gpuAllocator) noexcept override {}
            void configurePlugin(const Dims *inputDims, int nbInputs, const Dims *outputDims, int nbOutputs,
                                 const DataType *inputTypes, const DataType *outputTypes, const bool *inputIsBroadcast,
                                 const bool *outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) noexcept override {
                assert(*inputTypes == DataType::kFLOAT && floatFormat == PluginFormat::kLINEAR);
                assert(nbInputs == 2);
                assert(nbOutputs == 1);
                assert(inputDims[0].nbDims == 3);
                assert(inputDims[1].nbDims == 3);
                assert(outputDims[0].nbDims == 3);
                H = inputDims[0].d[1];
                W = inputDims[0].d[2];
                // prepare for the inference function
                if (mParam.imgH == 0 || mParam.imgW == 0) {
                    mParam.imgH = inputDims[1].d[1];
                    mParam.imgW = inputDims[1].d[2];
                }
                if (mParam.stepH == 0 || mParam.stepW == 0) {
                    mParam.stepH = static_cast<float>(mParam.imgH) / H;
                    mParam.stepW = static_cast<float>(mParam.imgW) / W;
                }
            }
            void detachFromContext() noexcept override {}

        private:
            Weights copyToDevice(const void *hostData, size_t count) {
                void* deviceData = nullptr;
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

            PriorBoxParameters mParam{};
            bool mOwnsParamMemory{false};
            int numPriors{0}, H{0}, W{0};
            Weights minSize{}, maxSize{}, aspectRatios{}; // not learnable weights
            std::string mPluginNamespace;
        };

        class PriorBoxPluginCreator : public BaseCreator {
        public:
            PriorBoxPluginCreator() {
                mPluginAttributes.clear();
                mPluginAttributes.emplace_back(PluginField("minSize", nullptr, PluginFieldType::kFLOAT32, 1));
                mPluginAttributes.emplace_back(PluginField("maxSize", nullptr, PluginFieldType::kFLOAT32, 1));
                mPluginAttributes.emplace_back(PluginField("aspectRatios", nullptr, PluginFieldType::kFLOAT32, 1));
                mPluginAttributes.emplace_back(PluginField("flip", nullptr, PluginFieldType::kINT32, 1));
                mPluginAttributes.emplace_back(PluginField("clip", nullptr, PluginFieldType::kINT32, 1));
                mPluginAttributes.emplace_back(PluginField("variance", nullptr, PluginFieldType::kFLOAT32, 4));
                mPluginAttributes.emplace_back(PluginField("imgH", nullptr, PluginFieldType::kINT32, 1));
                mPluginAttributes.emplace_back(PluginField("imgW", nullptr, PluginFieldType::kINT32, 1));
                mPluginAttributes.emplace_back(PluginField("stepH", nullptr, PluginFieldType::kFLOAT32, 1));
                mPluginAttributes.emplace_back(PluginField("stepW", nullptr, PluginFieldType::kFLOAT32, 1));
                mPluginAttributes.emplace_back(PluginField("offset", nullptr, PluginFieldType::kFLOAT32, 1));
                mFC.nbFields = mPluginAttributes.size();
                mFC.fields = mPluginAttributes.data();
            }
            ~PriorBoxPluginCreator() override = default;
            const char *getPluginName() const noexcept override { return PRIOR_BOX_PLUGIN_NAME; }
            const char *getPluginVersion() const noexcept override { return PRIOR_BOX_PLUGIN_VERSION; }
            const PluginFieldCollection *getFieldNames() noexcept override { return &mFC; }
            IPluginV2Ext *createPlugin(const char *name, const PluginFieldCollection *fc) noexcept override {
                const PluginField* fields = fc->fields;
                PriorBoxParameters params{};
                for (int i = 0; i < fc->nbFields; ++i) {
                    const char* attrName = fields[i].name;
                    if (!strcmp(attrName, "minSize")) {
                        assert(fields[i].type == PluginFieldType::kFLOAT32);
                        int size = fields[i].length;
                        params.minSize = new float[size];
                        const auto* minS = static_cast<const float*>(fields[i].data);
                        for (int j = 0; j < size; j++) {
                            params.minSize[j] = *minS;
                            minS++;
                        }
                        params.numMinSize = size;
                    } else if (!strcmp(attrName, "maxSize")) {
                        assert(fields[i].type == PluginFieldType::kFLOAT32);
                        int size = fields[i].length;
                        params.maxSize = new float[size];
                        const auto* maxS = static_cast<const float*>(fields[i].data);
                        for (int j = 0; j < size; j++) {
                            params.maxSize[j] = *maxS;
                            maxS++;
                        }
                        params.numMaxSize = size;
                    } else if (!strcmp(attrName, "aspectRatios")) {
                        assert(fields[i].type == PluginFieldType::kFLOAT32);
                        int size = fields[i].length;
                        params.aspectRatios = new float[size];
                        const auto* aR = static_cast<const float*>(fields[i].data);
                        for (int j = 0; j < size; j++) {
                            params.aspectRatios[j] = *aR;
                            aR++;
                        }
                        params.numAspectRatios = size;
                    } else if (!strcmp(attrName, "variance")) {
                        assert(fields[i].type == PluginFieldType::kFLOAT32);
                        int size = fields[i].length;
                        const auto* lVar = static_cast<const float*>(fields[i].data);
                        for (int j = 0; j < size; j++) {
                            params.variance[j] = (*lVar);
                            lVar++;
                        }
                    } else if (!strcmp(attrName, "flip")) {
                        assert(fields[i].type == PluginFieldType::kINT32);
                        params.flip = *(static_cast<const bool*>(fields[i].data));
                    } else if (!strcmp(attrName, "clip")) {
                        assert(fields[i].type == PluginFieldType::kINT32);
                        params.clip = *(static_cast<const bool*>(fields[i].data));
                    } else if (!strcmp(attrName, "imgH")) {
                        assert(fields[i].type == PluginFieldType::kINT32);
                        params.imgH = *(static_cast<const int*>(fields[i].data));
                    } else if (!strcmp(attrName, "imgW")) {
                        assert(fields[i].type == PluginFieldType::kINT32);
                        params.imgW = *(static_cast<const int*>(fields[i].data));
                    } else if (!strcmp(attrName, "stepH")) {
                        assert(fields[i].type == PluginFieldType::kFLOAT32);
                        params.stepH = *(static_cast<const float*>(fields[i].data));
                    } else if (!strcmp(attrName, "stepW")) {
                        assert(fields[i].type == PluginFieldType::kFLOAT32);
                        params.stepW = *(static_cast<const float*>(fields[i].data));
                    } else if (!strcmp(attrName, "offset")) {
                        assert(fields[i].type == PluginFieldType::kFLOAT32);
                        params.offset = *(static_cast<const float*>(fields[i].data));
                    }
                }
                auto* plugin = new PriorBox(params);
                plugin->setPluginNamespace(mNamespace.c_str());
                return plugin;
            }

            IPluginV2Ext *deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept override {
                auto* plugin = new PriorBox(serialData, serialLength);
                plugin->setPluginNamespace(mNamespace.c_str());
                return plugin;
            }

        private:
            PluginFieldCollection mFC{};
            std::vector<PluginField> mPluginAttributes{};
        };
    } // namespace plugin
} // namespace nvinfer1

#endif // TRT_PRIOR_BOX_PLUGIN_H
