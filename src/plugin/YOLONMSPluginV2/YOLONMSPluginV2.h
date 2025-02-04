/*** 
 * @Author: xingwg
 * @Date: 2024-10-12 09:59:37
 * @LastEditTime: 2024-10-12 15:43:14
 * @FilePath: /dmnn2/src/plugin/YOLONMSPluginV2/YOLONMSPluginV2.h
 * @Description: 
 * @
 * @Copyright (c) 2024 by Chinasvt, All Rights Reserved. 
 */
#ifndef ALGORITHMS_YOLONMSPLUGINV2_H
#define ALGORITHMS_YOLONMSPLUGINV2_H

#include <string>
#include <vector>
#include "../common/kernels/kernel.h"
#include "../common/nmsUtils.h"
#include "../common/plugin.h"
#include "base_types.h"
#include "error_check.h"

struct YOLODetectionOutputV2Parameters {
    int num_classes{1};
    float nms_thresh{0.45f};
    float conf_thresh{0.01f};
    int num_pts{5};
    int keep_topK{200};
    int topK{400};
};

using namespace nvinfer1;
using namespace nvinfer1::plugin;

namespace nvinfer1 {
    namespace plugin {
        static const char *YOLO_NMS_PLUGIN_V2_VERSION{"1"};
        static const char *YOLO_NMS_PLUGIN_V2_NAME{"YOLO_NMS_V2_TRT"};
        static const char *YOLO_NMS_DYNAMIC_PLUGIN_V2_NAME{"YOLO_NMS_DYNAMIC_V2_TRT"};
        class YOLODetectionOutputV2 : public IPluginV2Ext {
        public:
            explicit YOLODetectionOutputV2(YOLODetectionOutputV2Parameters param)
                    : param(param) {}

            YOLODetectionOutputV2(YOLODetectionOutputV2Parameters param, int C1, int C2)
                    : param(param) , C1(C1) , C2(C2) {}

            YOLODetectionOutputV2(const void *data, size_t length) {
                const char *d = reinterpret_cast<const char *>(data);
                param = read<YOLODetectionOutputV2Parameters>(d);
                C1 = read<int>(d);
                C2 = read<int>(d);
                assert(d == data + length);
            }

            ~YOLODetectionOutputV2() override = default;

            int getNbOutputs() const noexcept override { return 1; }
            Dims getOutputDimensions(int index, const Dims *inputs, int nbInputDims) noexcept override {
                assert(nbInputDims == 1);
                assert(index == 0);
                return Dims3(1, param.keep_topK, 7 + 2*param.num_pts);
            }
            int initialize() noexcept override { return STATUS_SUCCESS; }
            void terminate() noexcept override {}
            size_t getWorkspaceSize(int maxBatchSize) const noexcept override {
                return detectionInferenceWorkspaceSize(
                        maxBatchSize, C1, C2, param.num_classes, param.num_pts, param.topK, DataType::kFLOAT, DataType::kFLOAT);
            }
            int enqueue(int batchSize, void const* const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept override;
            size_t getSerializationSize() const noexcept override { return sizeof(YOLODetectionOutputV2Parameters) + sizeof(int) * 2; }
            void serialize(void *buffer) const noexcept override {
                char *d = reinterpret_cast<char *>(buffer);
                write(d, param);
                write(d, C1);
                write(d, C2);
                assert(d == buffer + getSerializationSize());
            }

            bool supportsFormat(DataType type, PluginFormat format) const noexcept override { return (type == DataType::kFLOAT && format == PluginFormat::kLINEAR); }
            const char *getPluginType() const noexcept override { return YOLO_NMS_PLUGIN_V2_NAME; }
            const char *getPluginVersion() const noexcept override { return YOLO_NMS_PLUGIN_V2_VERSION; }
            void destroy() noexcept override { delete this; }
            IPluginV2Ext *clone() const noexcept override {
                auto *plugin = new YOLODetectionOutputV2(param, C1, C2);
                plugin->setPluginNamespace(mPluginNamespace.c_str());
                return plugin;
            }
            void setPluginNamespace(const char *pluginNamespace) noexcept override { mPluginNamespace = pluginNamespace; }
            const char *getPluginNamespace() const noexcept override { return mPluginNamespace.c_str(); }
            DataType getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const noexcept override { return DataType::kFLOAT; }
            bool isOutputBroadcastAcrossBatch(int outputIndex, const bool *inputIsBroadcasted, int nbInputs) const noexcept override { return false; }
            bool canBroadcastInputAcrossBatch(int inputIndex) const noexcept override { return false; }
            void attachToContext(cudnnContext *cudnnContext, cublasContext *cublasContext, IGpuAllocator *gpuAllocator) noexcept override {}
            void detachFromContext() noexcept override {}
            void configurePlugin(const Dims *inputDims, int nbInputs, const Dims *outputDims, int nbOutputs,
                                 const DataType *inputTypes, const DataType *outputTypes, const bool *inputIsBroadcast,
                                 const bool *outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) noexcept override {

                assert(nbInputs == 1);
                assert(nbOutputs == 1);
                for (int i = 0; i < nbInputs; i++) {
                    if (inputDims[i].nbDims != 3 && inputDims[i].nbDims != 2)
                        LOG_FATAL("YOLOXNMSPlugin only supports 3D or 2D input!");
                }
                for (int i = 0; i < nbOutputs; i++) {
                    if (outputDims[i].nbDims != 3 && outputDims[i].nbDims != 2)
                        LOG_FATAL("YOLOXNMSPlugin only supports 3D or 2D output!");
                }
                C1 = inputDims[0].d[0]; // 80*80*3 + 40*40*3 + 20*20*3
                C2 = inputDims[0].d[1]; // 5 + num_classes + 2 * num_pts
                assert(C2 == 5 + param.num_classes + 2*param.num_pts);
            }

        private:
            YOLODetectionOutputV2Parameters param{};
            int C1{}, C2{};
            std::string mPluginNamespace;
        };

        class YOLODetectionOutputDynamicV2 : public IPluginV2DynamicExt {
        public:
            explicit YOLODetectionOutputDynamicV2(YOLODetectionOutputV2Parameters param)
                    : param(param) {}

            YOLODetectionOutputDynamicV2(YOLODetectionOutputV2Parameters param, int C1, int C2)
                    : param(param) , C1(C1) , C2(C2) {}

            YOLODetectionOutputDynamicV2(const void *data, size_t length) {
                const char *d = reinterpret_cast<const char *>(data);
                param = read<YOLODetectionOutputV2Parameters>(d);
                C1 = read<int>(d);
                C2 = read<int>(d);
                assert(d == data + length);
            }

            ~YOLODetectionOutputDynamicV2() override = default;

            int getNbOutputs() const noexcept override { return 1; }
            DimsExprs getOutputDimensions(int32_t outputIndex, nvinfer1::DimsExprs const* inputs, int32_t nbInputs, nvinfer1::IExprBuilder &exprBuilder) noexcept override {
                assert(nbInputs == 1);
                assert(outputIndex == 0 && outputIndex < this->getNbOutputs());
                assert(inputs[0].nbDims == 3);
                assert(!inputs[0].d[0]->isConstant());
                assert(inputs[0].d[1]->isConstant());
                assert(inputs[0].d[2]->isConstant());
                DimsExprs output{};
                output.nbDims = 4;
                output.d[0] = inputs[0].d[0];
                output.d[1] = exprBuilder.constant(1);
                output.d[2] = exprBuilder.constant(param.keep_topK);
                output.d[3] = exprBuilder.constant(7);
                return output;
            }
            int initialize() noexcept override { return STATUS_SUCCESS; }
            void terminate() noexcept override {}
            size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs, nvinfer1::PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override {
                return detectionInferenceWorkspaceSize(inputs[0].dims.d[0], C1, C2, param.num_classes, param.num_pts, param.topK, DataType::kFLOAT, DataType::kFLOAT);
            }
            int32_t enqueue(nvinfer1::PluginTensorDesc const *inputDesc,
                            nvinfer1::PluginTensorDesc const *outputDesc,
                            void const *const *inputs, void *const *outputs,
                            void *workspace, cudaStream_t stream) noexcept override;
            size_t getSerializationSize() const noexcept override { return sizeof(YOLODetectionOutputV2Parameters) + sizeof(int) * 2; }
            void serialize(void *buffer) const noexcept override {
                char *d = reinterpret_cast<char *>(buffer), *a = d;
                write(d, param);
                write(d, C1);
                write(d, C2);
                assert(d == a + getSerializationSize());
            }
            bool supportsFormatCombination(int32_t pos, const nvinfer1::PluginTensorDesc *inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override {
                assert(pos >= 0 && pos < 2);
                assert(nbInputs == 1);
                return inOut[pos].type == DataType::kFLOAT && inOut[pos].format == PluginFormat::kLINEAR;
            }
            const char *getPluginType() const noexcept override { return YOLO_NMS_DYNAMIC_PLUGIN_V2_NAME; }
            const char *getPluginVersion() const noexcept override { return YOLO_NMS_PLUGIN_V2_VERSION; }
            void destroy() noexcept override { delete this; }
            IPluginV2DynamicExt *clone() const noexcept override {
                auto *plugin = new YOLODetectionOutputDynamicV2(param, C1, C2);
                plugin->setPluginNamespace(mPluginNamespace.c_str());
                return plugin;
            }
            void setPluginNamespace(const char *pluginNamespace) noexcept override { mPluginNamespace = pluginNamespace; }
            const char *getPluginNamespace() const noexcept override { return mPluginNamespace.c_str(); }
            DataType getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const noexcept override { return DataType::kFLOAT; }
            void attachToContext(cudnnContext *cudnnContext, cublasContext *cublasContext, IGpuAllocator *gpuAllocator) noexcept override {}
            void detachFromContext() noexcept override {}
            void configurePlugin(nvinfer1::DynamicPluginTensorDesc const *in, int32_t nbInputs,
                                 nvinfer1::DynamicPluginTensorDesc const *out, int32_t nbOutputs) noexcept override {
                assert(nbInputs == 1);
                assert(nbOutputs == 1);
                assert(in[0].desc.dims.nbDims == 3);
                assert(out[0].desc.dims.nbDims == 4);
                C1 = in[0].desc.dims.d[1]; // 80*80*3 + 40*40*3 + 20*20*3
                C2 = in[0].desc.dims.d[2]; // 5 + num_classes + 2 * num_pts
                assert(C2 == 5 + param.num_classes + 2*param.num_pts);
            }
        private:
            YOLODetectionOutputV2Parameters param{};
            int C1{}, C2{};
            std::string mPluginNamespace;
        };

        class YOLONMSPluginV2Creator : public BaseCreator {
        public:
            YOLONMSPluginV2Creator() {
                mPluginAttributes.clear();
                mPluginAttributes.emplace_back(PluginField("conf_thresh", nullptr, PluginFieldType::kFLOAT32, 1));
                mPluginAttributes.emplace_back(PluginField("nms_thresh", nullptr, PluginFieldType::kFLOAT32, 1));
                mPluginAttributes.emplace_back(PluginField("nms_pts", nullptr, PluginFieldType::kINT32, 1));
                mPluginAttributes.emplace_back(PluginField("keep_top_k", nullptr, PluginFieldType::kINT32, 1));
                mPluginAttributes.emplace_back(PluginField("top_k", nullptr, PluginFieldType::kINT32, 1));
                mPluginAttributes.emplace_back(PluginField("num_classes", nullptr, PluginFieldType::kINT32, 1));
                mFC.nbFields = mPluginAttributes.size();
                mFC.fields = mPluginAttributes.data();
            }

            ~YOLONMSPluginV2Creator() override = default;
            const char *getPluginName() const noexcept override { return YOLO_NMS_PLUGIN_V2_NAME; }
            const char *getPluginVersion() const noexcept override { return YOLO_NMS_PLUGIN_V2_VERSION; }
            const PluginFieldCollection *getFieldNames() noexcept override { return &mFC; }
            IPluginV2Ext *createPlugin(const char *name, const PluginFieldCollection *fc) noexcept override {
                const PluginField *fields = fc->fields;
                YOLODetectionOutputV2Parameters params{};
                for (int i = 0; i < fc->nbFields; ++i) {
                    const char *attrName = fields[i].name;
                    if (!strcmp(attrName, "conf_thresh")) {
                        assert(fields[i].type == PluginFieldType::kFLOAT32);
                        params.conf_thresh = *(static_cast<const float *>(fields[i].data));
                    } else if (!strcmp(attrName, "nms_thresh")) {
                        assert(fields[i].type == PluginFieldType::kFLOAT32);
                        params.nms_thresh = *(static_cast<const float *>(fields[i].data));
                    } else if (!strcmp(attrName, "nms_pts")) {
                        assert(fields[i].type == PluginFieldType::kINT32);
                        params.num_pts = *(static_cast<const int *>(fields[i].data));
                    } else if (!strcmp(attrName, "keep_topK")) {
                        assert(fields[i].type == PluginFieldType::kINT32);
                        params.keep_topK = *(static_cast<const int *>(fields[i].data));
                    } else if (!strcmp(attrName, "topK")) {
                        assert(fields[i].type == PluginFieldType::kINT32);
                        params.topK = *(static_cast<const int *>(fields[i].data));
                    } else if (!strcmp(attrName, "num_classes")) {
                        assert(fields[i].type == PluginFieldType::kINT32);
                        params.num_classes = *(static_cast<const int *>(fields[i].data));
                    }
                }
                auto *plugin = new YOLODetectionOutputV2(params);
                plugin->setPluginNamespace(mNamespace.c_str());
                return plugin;
            }
            IPluginV2Ext *deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept override {
                auto *plugin = new YOLODetectionOutputV2(serialData, serialLength);
                plugin->setPluginNamespace(mNamespace.c_str());
                return plugin;
            }

        private:
            PluginFieldCollection mFC{};
            std::vector<PluginField> mPluginAttributes{};
        };

        class YOLONMSDynamicPluginV2Creator : public BaseCreator {
        public:
            YOLONMSDynamicPluginV2Creator() {
                mPluginAttributes.clear();
                mPluginAttributes.emplace_back(PluginField("conf_thresh", nullptr, PluginFieldType::kFLOAT32, 1));
                mPluginAttributes.emplace_back(PluginField("nms_thresh", nullptr, PluginFieldType::kFLOAT32, 1));
                mPluginAttributes.emplace_back(PluginField("nms_pts", nullptr, PluginFieldType::kINT32, 1));
                mPluginAttributes.emplace_back(PluginField("keep_top_k", nullptr, PluginFieldType::kINT32, 1));
                mPluginAttributes.emplace_back(PluginField("top_k", nullptr, PluginFieldType::kINT32, 1));
                mPluginAttributes.emplace_back(PluginField("num_classes", nullptr, PluginFieldType::kINT32, 1));
                mFC.nbFields = mPluginAttributes.size();
                mFC.fields = mPluginAttributes.data();
            }
            ~YOLONMSDynamicPluginV2Creator() override = default;
            const char *getPluginName() const noexcept override { return YOLO_NMS_DYNAMIC_PLUGIN_V2_NAME; }
            const char *getPluginVersion() const noexcept override { return YOLO_NMS_PLUGIN_V2_VERSION; }
            const PluginFieldCollection *getFieldNames() noexcept override { return &mFC; }
            IPluginV2DynamicExt *createPlugin(const char *name, const PluginFieldCollection *fc) noexcept override {
                const PluginField *fields = fc->fields;
                YOLODetectionOutputV2Parameters params{};
                for (int i = 0; i < fc->nbFields; ++i) {
                    const char *attrName = fields[i].name;
                    if (!strcmp(attrName, "conf_thresh")) {
                        assert(fields[i].type == PluginFieldType::kFLOAT32);
                        params.conf_thresh = *(static_cast<const float *>(fields[i].data));
                    } else if (!strcmp(attrName, "nms_thresh")) {
                        assert(fields[i].type == PluginFieldType::kFLOAT32);
                        params.nms_thresh = *(static_cast<const float *>(fields[i].data));
                    } else if (!strcmp(attrName, "nms_pts")) {
                        assert(fields[i].type == PluginFieldType::kINT32);
                        params.num_pts = *(static_cast<const int *>(fields[i].data));
                    } else if (!strcmp(attrName, "keep_topK")) {
                        assert(fields[i].type == PluginFieldType::kINT32);
                        params.keep_topK = *(static_cast<const int *>(fields[i].data));
                    } else if (!strcmp(attrName, "topK")) {
                        assert(fields[i].type == PluginFieldType::kINT32);
                        params.topK = *(static_cast<const int *>(fields[i].data));
                    } else if (!strcmp(attrName, "num_classes")) {
                        assert(fields[i].type == PluginFieldType::kINT32);
                        params.num_classes = *(static_cast<const int *>(fields[i].data));
                    }
                }
                auto *plugin = new YOLODetectionOutputDynamicV2(params);
                plugin->setPluginNamespace(mNamespace.c_str());
                return plugin;
            }
            IPluginV2DynamicExt *deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept override {
                auto *plugin = new YOLODetectionOutputDynamicV2(serialData, serialLength);
                plugin->setPluginNamespace(mNamespace.c_str());
                return plugin;
            }
        private:
            PluginFieldCollection mFC{};
            std::vector<PluginField> mPluginAttributes{};
        };
    } // namespace plugin
} // namespace nvinfer1

#endif //ALGORITHMS_YOLONMSPLUGINV2_H
