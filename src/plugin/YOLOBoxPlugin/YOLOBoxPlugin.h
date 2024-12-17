/*** 
 * @Author: xingwg
 * @Date: 2024-10-12 09:59:37
 * @LastEditTime: 2024-10-12 12:09:14
 * @FilePath: /dmnn2/src/plugin/YOLOBoxPlugin/YOLOBoxPlugin.h
 * @Description: 
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved. 
 */
#ifndef ALGORITHMS_YOLOBOXPLUGIN_H
#define ALGORITHMS_YOLOBOXPLUGIN_H

#include <string>
#include <vector>
#include <cstdlib>
#include <cublas_v2.h>
#include <cudnn.h>
#include "../common/kernels/kernel.h"
#include "../common/plugin.h"
#include "error_check.h"
#include "base_types.h"

struct YoloBoxParameters {
    float *anchor{nullptr};
    float stride{0};
    int num_anchors{0};
};

namespace nvinfer1 {
    namespace plugin {
        static const char *YOLO_BOX_PLUGIN_VERSION{"1"};
        static const char *YOLO_BOX_PLUGIN_NAME{"YoloBox_TRT"};
        class YOLOBox : public IPluginV2Ext {
        public:
            explicit YOLOBox(YoloBoxParameters param)
                : mParam_(param), mOwnsParamMemory_(true) {
                assert(param.anchor != nullptr);
                for (int i = 0; i < param.num_anchors; ++i)
                    assert(param.anchor[i] > 0 && "anchor_x must be positive");
                mAnchors_ = copyToDevice(param.anchor, param.num_anchors);
                assert(param.stride > 0 && "yolo stride must be positive");
            }
            YOLOBox(YoloBoxParameters param, int H, int W) {}
            YOLOBox(YoloBoxParameters param, int C, int H, int W, Weights anchors)
                : mParam_(param), mOwnsParamMemory_(false), C_(C), H_(H), W_(W), mAnchors_(anchors) {}
            YOLOBox(const void *buffer, size_t length)
                : mOwnsParamMemory_(true) {
                const char *d = reinterpret_cast<const char *>(buffer);
                mParam_ = read<YoloBoxParameters>(d);
                mParam_.anchor = new float[mParam_.num_anchors];
                C_ = read<int>(d);  // 3
                H_ = read<int>(d);  // 6400/1600/400
                W_ = read<int>(d);  // 9
                for (auto i = 0; i < mParam_.num_anchors; i++)
                    mParam_.anchor[i] = reinterpret_cast<const float*>(d)[i];
                mAnchors_ = deserializeToDevice(d, mParam_.num_anchors);
                assert(d == buffer + length);
            }
            ~YOLOBox() override = default;
            int getNbOutputs() const noexcept override { return 1; }
            Dims getOutputDimensions(int index, const Dims *inputs, int nbInputDims) noexcept override {
                assert(nbInputDims == 2);
                assert(index == 0);
                C_ = inputs[0].d[0];
                H_ = inputs[0].d[1];
                W_ = inputs[0].d[2]; // 9
                return Dims3(C_ * H_, W_, 1);  // [3*6400, 9, 1] [3*1600, 9, 1] [3*400, 9, 1]
            }
            int initialize() noexcept override { return STATUS_SUCCESS; }
            void terminate() noexcept override {
                if (mOwnsParamMemory_) {
                    CUDACHECK(cudaFree(const_cast<void*>(mAnchors_.values)));
                    delete[] mParam_.anchor;
                }
            }
            size_t getWorkspaceSize(int maxBatchSize) const noexcept override { return 0; }
            int enqueue(int32_t batchSize, void const *const *inputs, void *const *outputs, void *workspace,
                    cudaStream_t stream) noexcept override;
            size_t getSerializationSize() const noexcept override {
                return sizeof(YoloBoxParameters) + sizeof(int) * 3 + sizeof(float) * mParam_.num_anchors;
            }
            void serialize(void *buffer) const noexcept override {
                char *d = reinterpret_cast<char *>(buffer);
                write(d, mParam_);
                write(d, C_);
                write(d, H_);
                write(d, W_);
                serializeFromDevice(d, mAnchors_);
                assert(d == buffer + getSerializationSize());
            }
            bool supportsFormat(DataType type, PluginFormat format) const noexcept override {
                return (type == DataType::kFLOAT && format == PluginFormat::kLINEAR);
            }
            const char *getPluginType() const noexcept override { return YOLO_BOX_PLUGIN_NAME; }
            const char *getPluginVersion() const noexcept override { return YOLO_BOX_PLUGIN_VERSION; }
            void destroy() noexcept override { delete this; }
            IPluginV2Ext *clone() const noexcept override {
                auto *plugin = new YOLOBox(mParam_, C_, H_, W_, mAnchors_);
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
                                 const bool *outputIsBroadcast, PluginFormat floatFormat,
                                 int maxBatchSize) noexcept override {
                assert(*inputTypes == DataType::kFLOAT && floatFormat == PluginFormat::kLINEAR);
                assert(nbInputs == 2);
                assert(nbOutputs == 1);
                assert(inputDims[0].nbDims == 3);
                assert(inputDims[1].nbDims == 3);
                assert(outputDims[0].nbDims == 3);
                C_ = inputDims[0].d[0];
                H_ = inputDims[0].d[1];
                W_ = inputDims[0].d[2];
            }
            void detachFromContext() noexcept override {}

        private:
            Weights copyToDevice(const void *hostData, size_t count) {
                void *deviceData = nullptr;
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
            YoloBoxParameters mParam_;
            bool mOwnsParamMemory_;
            int H_, W_, C_;
            Weights mAnchors_;
            std::string mPluginNamespace;
        };

        class YOLOBoxPluginCreator : public BaseCreator {
        public:
            YOLOBoxPluginCreator() {
                mPluginAttributes.clear();
                mPluginAttributes.emplace_back(PluginField("anchor", nullptr, PluginFieldType::kFLOAT32, 1));
                mPluginAttributes.emplace_back(PluginField("stride", nullptr, PluginFieldType::kFLOAT32, 1));
                mFC.nbFields = mPluginAttributes.size();
                mFC.fields = mPluginAttributes.data();
            }
            ~YOLOBoxPluginCreator() override = default;
            const char *getPluginName() const noexcept override { return YOLO_BOX_PLUGIN_NAME; }
            const char *getPluginVersion() const noexcept override { return YOLO_BOX_PLUGIN_VERSION; }
            const PluginFieldCollection *getFieldNames() noexcept override { return &mFC; }
            IPluginV2Ext *createPlugin(const char *name, const PluginFieldCollection *fc) noexcept override {
                const PluginField *fields = fc->fields;
                YoloBoxParameters params;
                for (int i = 0; i < fc->nbFields; ++i) {
                    const char *attrName = fields[i].name;
                    if (!strcmp(attrName, "anchor")) {
                        assert(fields[i].type == PluginFieldType::kFLOAT32);
                        int size = fields[i].length;
                        params.anchor = new float[size];
                        params.num_anchors = size;
                        const auto *anchor = static_cast<const float*>(fields[i].data);
                        for (int j = 0; j < size; j++)
                            params.anchor[j] = anchor[j];
                    } else if (!strcmp(attrName, "stride")) {
                        assert(fields[i].type == PluginFieldType::kFLOAT32);
                        params.stride = *(static_cast<const float *>(fields[i].data));
                    }
                }
                auto *plugin = new YOLOBox(params);
                plugin->setPluginNamespace(mNamespace.c_str());
                return plugin;
            }
            IPluginV2Ext *deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept override {
                auto *plugin = new YOLOBox(serialData, serialLength);
                plugin->setPluginNamespace(mNamespace.c_str());
                return plugin;
            }

        private:
            PluginFieldCollection mFC{};
            std::vector<PluginField> mPluginAttributes{};
        };
    } // namespace plugin
} // namespace nvinfer1

#endif //ALGORITHMS_YOLOBOXPLUGIN_H
